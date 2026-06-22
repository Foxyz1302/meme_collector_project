/// Downloads all model files needed by the app.
///
/// Run from the workspace root:
///   dart run scripts/download_models.dart
///
/// Downloads to: assets/models/
/// Total size: ~478 MB
///
/// Files downloaded:
///   potion-base-32M/                    (~150 MB)  text embeddings
///   clip_text_int8.onnx                 (64 MB)   CLIP text tower
///   clip_vision_int8.onnx               (89 MB)   CLIP vision tower
///   clip_tokenizer.json                 (~1 MB)   BPE tokenizer spec
///   ppocr_det.onnx                      (84 MB)   PP-OCRv5 detection
///   ppocr_rec.onnx                      (~10 MB)  PP-OCRv5 recognition
///   ppocr_dict.txt                      (~50 KB)  character dictionary
///
/// Source: Hugging Face Hub (no auth required for these public models).

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;

const _hfBase = 'https://huggingface.co';

final _downloads = <_Download>[
  // ─── model2vec: potion-base-32M ─────────────────────────────────────────
  _Download(
    url: '$_hfBase/minishlab/potion-base-32M/resolve/main/model.safetensors',
    dest: 'potion-base-32M/model.safetensors',
    expectedSizeBytes: 130 * 1024 * 1024, // ~130 MB
  ),
  _Download(
    url: '$_hfBase/minishlab/potion-base-32M/resolve/main/tokenizer.json',
    dest: 'potion-base-32M/tokenizer.json',
  ),
  _Download(
    url: '$_hfBase/minishlab/potion-base-32M/resolve/main/tokenizer_config.json',
    dest: 'potion-base-32M/tokenizer_config.json',
  ),
  _Download(
    url: '$_hfBase/minishlab/potion-base-32M/resolve/main/special_tokens_map.json',
    dest: 'potion-base-32M/special_tokens_map.json',
  ),
  _Download(
    url: '$_hfBase/minishlab/potion-base-32M/resolve/main/config.json',
    dest: 'potion-base-32M/config.json',
  ),

  // ─── CLIP ViT-B/32 INT8 (Xenova split towers) ──────────────────────────
  _Download(
    url: '$_hfBase/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_int8.onnx',
    dest: 'clip_text_int8.onnx',
    expectedSizeBytes: 64 * 1024 * 1024,
  ),
  _Download(
    url: '$_hfBase/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_int8.onnx',
    dest: 'clip_vision_int8.onnx',
    expectedSizeBytes: 89 * 1024 * 1024,
  ),
  _Download(
    url: '$_hfBase/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json',
    dest: 'clip_tokenizer.json',
  ),

  // ─── PP-OCRv5 (monkt/paddleocr-onnx) ───────────────────────────────────
  _Download(
    url: '$_hfBase/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx',
    dest: 'ppocr_det.onnx',
    expectedSizeBytes: 84 * 1024 * 1024,
  ),
  // English recognition model — lives under languages/english/, not recognition/
  _Download(
    url: '$_hfBase/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx',
    dest: 'ppocr_rec.onnx',
    expectedSizeBytes: 7 * 1024 * 1024,
  ),
  // English character dictionary — lives next to rec.onnx
  _Download(
    url: '$_hfBase/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt',
    dest: 'ppocr_dict.txt',
  ),
];

// No fallbacks needed — paths verified from the monkt README:
// https://huggingface.co/monkt/paddleocr-onnx/blob/main/README.md
// All v5 recognition models live under languages/{language}/rec.onnx + dict.txt

class _Download {
  final String url;
  final String dest;
  final int? expectedSizeBytes;
  final bool optional;

  const _Download({
    required this.url,
    required this.dest,
    this.expectedSizeBytes,
    this.optional = false,
  });
}

Future<void> main() async {
  final destRoot = p.join(Directory.current.path, 'assets', 'models');
  print('Downloading models to: $destRoot\n');

  await Directory(destRoot).create(recursive: true);
  final dio = Dio();

  var successCount = 0;
  var failCount = 0;

  for (final dl in _downloads) {
    final destPath = p.join(destRoot, dl.dest);
    await Directory(p.dirname(destPath)).create(recursive: true);

    // Skip if already downloaded and size matches
    final existingFile = File(destPath);
    if (await existingFile.exists()) {
      final size = await existingFile.length();
      if (dl.expectedSizeBytes == null || size >= dl.expectedSizeBytes! * 0.9) {
        print('[SKIP] ${dl.dest} (already exists, ${(size / 1024 / 1024).toStringAsFixed(1)} MB)');
        successCount++;
        continue;
      }
    }

    print('[DOWN] ${dl.dest} from ${dl.url}');
    try {
      await dio.download(
        dl.url,
        destPath,
        onReceiveProgress: (received, total) {
          if (total > 0) {
            final pct = (received / total * 100).toStringAsFixed(1);
            final recvMb = (received / 1024 / 1024).toStringAsFixed(1);
            final totMb = (total / 1024 / 1024).toStringAsFixed(1);
            stdout.write('\r        $recvMb / $totMb MB ($pct%)    ');
          }
        },
      );
      stdout.write('\n');
      print('        OK');
      successCount++;
    } on DioException catch (e) {
      stdout.write('\n');
      if (dl.optional) {
        print('        FAILED (optional): ${e.message}');
        failCount++;
      } else {
        print('        FAILED: ${e.message}');
        failCount++;
      }
    }
  }

  print('\n========================================');
  print('Downloaded: $successCount succeeded, $failCount failed');
  print('Location:   $destRoot');

  // Print final size
  try {
    var totalSize = 0;
    await for (final entry
        in Directory(destRoot).list(recursive: true, followLinks: false)) {
      if (entry is File) {
        totalSize += await entry.length();
      }
    }
    print('Total size: ${(totalSize / 1024 / 1024).toStringAsFixed(1)} MB');
  } catch (_) {}

  if (failCount > 0) {
    print('\nNote: Some optional files failed. PP-OCR may need manual setup.');
    print('See: https://huggingface.co/monkt/paddleocr-onnx');
    exit(1);
  }
  print('\nAll downloads complete. Run validate_onnx_models.dart next.');
}
