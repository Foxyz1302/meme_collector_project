/// Concrete InferenceFactory implementation for the Windows app.
///
/// Creates the ONNX + model2vec embedders that the Coordinator uses.
/// This is the bridge between the pure-Dart core package and the
/// Flutter-dependent ONNX code.

import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:path/path.dart' as p;

import 'inference_impl.dart';

/// Factory that creates concrete embedder implementations for the app.
///
/// All model paths are resolved relative to the assets/models/ directory.
/// On Windows desktop, assets live at <workspace_root>/assets/models/.
class AppInferenceFactory implements InferenceFactory {
  final String modelsDir;

  AppInferenceFactory({required this.modelsDir});

  // ─── Paths ────────────────────────────────────────────────────────────

  String get _model2vecPath => p.join(modelsDir, 'potion-base-32M');
  String get _clipTextPath => p.join(modelsDir, 'clip_text_fp16.onnx');
  String get _clipVisionPath => p.join(modelsDir, 'clip_vision_fp16.onnx');
  String get _clipTokenizerPath => p.join(modelsDir, 'clip_tokenizer.json');
  String get _ppocrDetPath => p.join(modelsDir, 'ppocr_det.onnx');
  String get _ppocrRecPath => p.join(modelsDir, 'ppocr_rec.onnx');
  String get _ppocrDictPath => p.join(modelsDir, 'ppocr_dict.txt');

  // ─── InferenceFactory implementation ──────────────────────────────────

  @override
  TextEmbedder createQueryEmbedder() {
    // model2vec — fast, used for every search query (~1ms)
    return Model2VecEmbedder(modelPath: _model2vecPath);
  }

  @override
  TextEmbedder? createClipTextEmbedder() {
    // CLIP text tower — used for cross-modal search (text → image)
    // Optional: returns null if you want text-only search
    return ClipTextEmbedder(
      modelPath: _clipTextPath,
      tokenizerPath: _clipTokenizerPath,
    );
  }

  @override
  TextEmbedder createIngestTextEmbedder() {
    // Same as query embedder — model2vec is fast enough to share
    return Model2VecEmbedder(modelPath: _model2vecPath);
  }

  @override
  ImageEmbedder? createImageEmbedder() {
    // CLIP vision tower — used at ingest time to embed images
    return ClipImageEmbedder(modelPath: _clipVisionPath);
  }

  @override
  OcrEngine? createOcrEngine() {
    // PP-OCRv5 — used at ingest time to extract overlaid meme text
    // TODO: implement PpocrEngine (det + rec pipeline) — currently returns null
    // Once implemented, this returns PpocrEngine(
    //   detModelPath: _ppocrDetPath,
    //   recModelPath: _ppocrRecPath,
    //   dictPath: _ppocrDictPath,
    // );
    return null;
  }
}
