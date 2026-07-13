import sys
from io import BytesIO
from types import ModuleType

import numpy as np
from PIL import Image

from base.config import load_config
from core.rag.utils.document_loaders import _image_bytes_to_array, _ocr_lines
from core.rag.utils.text_splitters import (
    MODEL_PATH,
    AliTextSplitter,
    ChineseRecursiveTextSplitter,
    _split_text_with_regex_from_end,
)


def test_default_modelscope_model_uses_full_model_id():
    model_id = MODEL_PATH["segment_model"]["ali_model"]

    assert model_id == "iic/nlp_bert_document-segmentation_chinese-base"


def test_regex_splitter_keeps_chinese_and_english_punctuation_at_end():
    chinese = _split_text_with_regex_from_end(
        "甲。乙！丙？丁",
        r"[。！？]",
        True,
    )
    english = _split_text_with_regex_from_end(
        "First. Second! Third? Last",
        r"[.!?]\s",
        True,
    )

    assert chinese == ["甲。", "乙！", "丙？", "丁"]
    assert english == ["First. ", "Second! ", "Third? ", "Last"]


def test_chinese_splitter_does_not_insert_regex_as_text():
    splitter = ChineseRecursiveTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
        keep_separator=False,
    )

    assert splitter.split_text("甲。乙！丙") == ["甲乙丙"]


def test_chinese_splitter_falls_back_to_character_boundaries():
    splitter = ChineseRecursiveTextSplitter(
        chunk_size=4,
        chunk_overlap=0,
    )

    assert splitter.split_text("甲乙丙丁戊己庚辛") == ["甲乙丙丁", "戊己庚辛"]


def test_ali_splitter_uses_modelscope_segment_boundaries(monkeypatch):
    pipeline_module = ModuleType("modelscope.pipelines")
    pipeline_options = {}

    def create_pipeline(**kwargs):
        pipeline_options.update(kwargs)
        return lambda **payload: {"text": "第一段\n\t第二段"}

    pipeline_module.pipeline = create_pipeline
    monkeypatch.setitem(sys.modules, "modelscope.pipelines", pipeline_module)

    config = load_config()
    splitter = AliTextSplitter.from_config(
        config,
        chunk_size=100,
        chunk_overlap=0,
    )

    assert splitter.split_text("原始文本") == ["第一段", "第二段"]
    assert pipeline_options == {
        "task": "document-segmentation",
        "model": "iic/nlp_bert_document-segmentation_chinese-base",
        "device": config.rag.segmenter_device,
    }


def test_document_loader_helpers_normalize_images_and_ocr_text():
    image_buffer = BytesIO()
    Image.new("L", (3, 2), color=255).save(image_buffer, format="PNG")

    image_array = _image_bytes_to_array(image_buffer.getvalue())
    ocr = lambda image: ([[None, "图片文字", 0.99]], None)

    assert image_array.shape == (2, 3, 3)
    assert image_array.dtype == np.uint8
    assert _ocr_lines(ocr, image_array) == ["图片文字"]
