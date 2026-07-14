from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from base.config import AppConfig, load_config
from base.logger import logger
from core.rag.constants import (
    DEFAULT_MODEL_DEVICE,
    DEFAULT_QUERY_MODEL,
    DEFAULT_QUERY_MODEL_PATH,
    DEFAULT_QUERY_TRAINING_DATA_PATH,
    GENERAL_KNOWLEDGE_CATEGORY,
    PROFESSIONAL_CONSULTATION_CATEGORY,
    QUERY_CATEGORY_LOG_NAMES,
    QUERY_LABEL_MAP,
    normalize_query_category,
)
from core.rag.prompt import RAGPrompts


log = logger.bind(module=__name__)


class QueryDataset(Dataset):
    def __init__(
        self,
        encodings: Mapping[str, Any],
        labels: Sequence[int],
    ) -> None:
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class QueryClassifier:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_QUERY_MODEL_PATH,
        *,
        base_model: str = DEFAULT_QUERY_MODEL,
        training_data_path: str | Path = DEFAULT_QUERY_TRAINING_DATA_PATH,
        tokenizer: Any | None = None,
        model: Any | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.training_data_path = Path(training_data_path)
        self.device = torch.device(device or DEFAULT_MODEL_DEVICE)
        self.label_map = dict(QUERY_LABEL_MAP)
        self.tokenizer = tokenizer or self._load_tokenizer()
        self.model = model

        if self.model is None:
            self.load_model()
        else:
            self._configure_model_metadata()
            self.model.to(self.device)
            log.info("Query classifier initialized with an injected model")
        log.info("Query classifier initialized: device={}", self.device)

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
        device: str | torch.device | None = None,
    ) -> "QueryClassifier":
        return cls(
            model_path=config.rag.query_model_path,
            base_model=config.rag.query_base_model,
            training_data_path=config.rag.query_training_data_path,
            tokenizer=tokenizer,
            model=model,
            device=device if device is not None else config.rag.model_device,
        )

    def _load_tokenizer(self) -> Any:
        source = self.model_path if self.model_path.exists() else self.base_model
        tokenizer = AutoTokenizer.from_pretrained(str(source))
        log.info("Loaded query tokenizer: source={}", source)
        return tokenizer

    def load_model(self) -> None:
        if self.model_path.exists():
            source = self.model_path
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(source)
            )
            log.info("Loaded query classifier model: path={}", source)
        else:
            source = self.base_model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                source,
                num_labels=len(self.label_map),
            )
            log.info("Initialized query classifier model: source={}", source)
        self._configure_model_metadata()
        self.model.to(self.device)

    def _configure_model_metadata(self) -> None:
        """Persist the classification contract without tokenizing it."""
        model_config = getattr(self.model, "config", None)
        if model_config is None:
            return
        model_config.id2label = {
            label_id: label
            for label, label_id in self.label_map.items()
        }
        model_config.label2id = dict(self.label_map)
        model_config.problem_type = "single_label_classification"
        model_config.query_classification_prompt = (
            RAGPrompts.query_classification_prompt().template
        )

    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("query classifier model is not initialized")
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        log.info("Saved query classifier model: path={}", self.model_path)

    def preprocess_data(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> tuple[Mapping[str, Any], list[int]]:
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        try:
            encoded_labels = [
                self.label_map[normalize_query_category(label)]
                for label in labels
            ]
        except KeyError as exc:
            raise ValueError(f"unsupported query label: {exc.args[0]}") from exc
        return encodings, encoded_labels

    def create_dataset(
        self,
        encodings: Mapping[str, Any],
        labels: Sequence[int],
    ) -> QueryDataset:
        return QueryDataset(encodings, labels)

    def train_model(
        self,
        data_file: str | Path | None = None,
    ) -> None:
        data_path = (
            Path(data_file)
            if data_file is not None
            else self.training_data_path
        )
        if not data_path.exists():
            log.error("Query training dataset not found: path={}", data_path)
            raise FileNotFoundError(f"query training dataset not found: {data_path}")

        with data_path.open("r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        if not records:
            raise ValueError("query training dataset is empty")

        texts = [record["query"] for record in records]
        labels = [record["label"] for record in records]
        train_texts, validation_texts, train_labels, validation_labels = (
            train_test_split(
                texts,
                labels,
                test_size=0.2,
                random_state=42,
            )
        )

        train_encodings, encoded_train_labels = self.preprocess_data(
            train_texts,
            train_labels,
        )
        validation_encodings, encoded_validation_labels = self.preprocess_data(
            validation_texts,
            validation_labels,
        )
        train_dataset = self.create_dataset(
            train_encodings,
            encoded_train_labels,
        )
        validation_dataset = self.create_dataset(
            validation_encodings,
            encoded_validation_labels,
        )

        training_args = TrainingArguments(
            output_dir="./bert_results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="./bert_logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
            use_cpu=self.device.type == "cpu",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=self.compute_metrics,
        )

        log.info("Query classifier training started: records={}", len(records))
        trainer.train()
        self.save_model()
        self.evaluate_model(validation_texts, encoded_validation_labels)
        log.info("Query classifier training completed")

    @staticmethod
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == labels).mean())}

    def evaluate_model(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> dict[str, Any]:
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        dataset = self.create_dataset(encodings, labels)
        predictions = Trainer(model=self.model).predict(dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        report = classification_report(
            labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["general_knowledge", "professional_consultation"],
            zero_division=0,
        )
        matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
        log.info("Query classifier evaluation report:\n{}", report)
        log.info("Query classifier confusion matrix:\n{}", matrix)
        return {"classification_report": report, "confusion_matrix": matrix}

    def predict_category(self, query: str) -> str:
        if self.model is None:
            log.error("Query classifier model is not initialized")
            return GENERAL_KNOWLEDGE_CATEGORY

        encoding = self.tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = int(torch.argmax(outputs.logits, dim=1).item())

        category = (
            PROFESSIONAL_CONSULTATION_CATEGORY
            if prediction == self.label_map[PROFESSIONAL_CONSULTATION_CATEGORY]
            else GENERAL_KNOWLEDGE_CATEGORY
        )
        log.info(
            "Query classified: category={}",
            QUERY_CATEGORY_LOG_NAMES[category],
        )
        return category
