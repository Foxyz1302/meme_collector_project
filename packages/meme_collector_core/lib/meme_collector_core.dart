/// meme_collector_core — backend for Reaction Roulette.
///
/// Public API. Import this file, not the individual parts.
///
/// ```dart
/// import 'package:meme_collector_core/meme_collector_core.dart';
/// ```
///
/// Source files (in dependency order):
///   - models.dart    — Reaction entity + JSON codec
///   - storage.dart   — metadata.json I/O + atomic rename + paths
///   - inference.dart — abstract TextEmbedder / ImageEmbedder / OcrEngine
///   - search.dart    — in-memory SIMD vector index + RRF fusion
///   - ingest.dart    — URL parser + ffmpeg wrapper + pipeline state machine
///   - worker.dart    — two isolates + SendPort message protocol + Coordinator
///
/// Core is pure Dart (no Flutter deps). The app provides concrete ONNX
/// implementations of the inference interfaces via InferenceFactory.
library meme_collector_core;

export 'models.dart';
export 'storage.dart';
export 'inference.dart';
export 'search.dart';
export 'ingest.dart';
export 'worker.dart';
