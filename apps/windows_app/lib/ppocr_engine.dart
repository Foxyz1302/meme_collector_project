/// PP-OCRv5 OCR engine — detection + recognition pipeline.
///
/// Two-stage pipeline:
///   1. Detection model (ppocr_det.onnx): image → probability map → text boxes
///   2. Recognition model (ppocr_rec.onnx): crop each box → character sequence
///
/// Models validated in integration_test/validate_models_test.dart:
///   - ppocr_det.onnx: input [1, 3, 960, 960], output [1, 1, 960, 960]
///   - ppocr_rec.onnx: input [1, 3, 48, 320], output [1, 40, 438]
///
/// The detection model outputs a per-pixel probability map where high values
/// indicate text. We threshold it, find bounding boxes via connected components
/// (simplified), crop each region, resize to 48×320, and run recognition.
///
/// The recognition model outputs [1, 40, 438] — 40 timesteps × 438 character
/// classes. We decode via CTC (take argmax per timestep, collapse repeats,
/// remove blanks).

library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

class PpocrEngine implements OcrEngine {
  final String _detModelPath;
  final String _recModelPath;
  final String _dictPath;

  OrtSession? _detSession;
  OrtSession? _recSession;
  List<String>? _dict;
  bool _initialized = false;
  int _blankIndex = 0;

  /// Detection model input size (960×960 is the default for PP-OCRv5)
  static const int _detSize = 960;

  /// Recognition model input size
  static const int _recHeight = 48;
  static const int _recWidth = 320;

  /// Threshold for text detection probability map
  static const double _detThreshold = 0.3;

  /// Blank token index for CTC decoding (last index in the dict)

  PpocrEngine({required String detModelPath, required String recModelPath, required String dictPath})
    : _detModelPath = detModelPath,
      _recModelPath = recModelPath,
      _dictPath = dictPath;

  @override
  bool get isLoaded => _initialized;

  @override
  Future<void> init() async {
    if (_initialized) return;

    final detBytes = await File(_detModelPath).readAsBytes();
    final recBytes = await File(_recModelPath).readAsBytes();

    final sessionOptions = OrtSessionOptions();
    sessionOptions.appendDefaultProviders();

    _detSession = OrtSession.fromBuffer(detBytes, sessionOptions);
    _recSession = OrtSession.fromBuffer(recBytes, sessionOptions);

    // Load character dictionary
    final dictContent = await File(_dictPath).readAsString();
    _dict = dictContent.split('\n').where((l) => l.isNotEmpty).toList();
    // PP-OCR uses CTC blank at index 0, not the last index.
    // The dict file contains the actual characters (no blank entry).
    // The model outputs num_classes = dict.length + 1 (extra for blank at 0).
    // So index 0 = blank, indices 1..N = dict[0]..dict[N-1].
    _blankIndex = 0;

    _initialized = true;
    print(
      '[OCR] Initialized: det=${_detSession!.inputNames}, '
      'rec=${_recSession!.inputNames}, dict=${_dict!.length} chars',
    );
  }

  @override
  Future<void> dispose() async {
    await _detSession?.release();
    await _recSession?.release();
    _detSession = null;
    _recSession = null;
    _dict = null;
    _initialized = false;
  }

  @override
  Future<String?> ocrFile(String imagePath) async {
    if (!_initialized) {
      throw StateError('PpocrEngine not initialized. Call init() first.');
    }

    try {
      // 1. Load + decode image
      final bytes = await File(imagePath).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return null;

      // 2. Run detection → get text boxes
      final boxes = await _detect(decoded);
      if (boxes.isEmpty) return null;

      print('[OCR] Detected ${boxes.length} text regions');

      // 3. Run recognition on each box
      final results = <String>[];
      for (final box in boxes) {
        final crop = _cropBox(decoded, box);
        if (crop.width < 5 || crop.height < 5) continue; // skip tiny regions
        final text = await _recognize(crop);
        if (text.isNotEmpty) results.add(text);
      }

      if (results.isEmpty) return null;
      return results.join(' ');
    } catch (e) {
      print('[OCR] Error: $e');
      return null;
    }
  }

  // ─── Detection ─────────────────────────────────────────────────────────

  /// Run the detection model and extract text region bounding boxes.
  Future<List<_TextBox>> _detect(img.Image image) async {
    // Resize to 960×960 (maintaining aspect ratio with padding)
    final resized = _resizeForDetection(image);

    // Convert to NCHW float32, normalize to [0, 1]
    final input = _imageToNchw(resized, _detSize, _detSize, normalize: true);

    // Run inference
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, _detSize, _detSize]);
    final runOptions = OrtRunOptions();
    final outputs = await _detSession!.run(runOptions, {'x': inputTensor});
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty || outputs.first == null) return [];

    // Output is [1, 1, 960, 960] probability map
    final outputValue = outputs.first!.value;
    outputs.first!.release();

    // Flatten and threshold
    final flat = _flattenToList(outputValue);
    if (flat.isEmpty) return [];

    // Find text regions via simple connected components
    final boxes = _findBoxes(flat, _detSize, _detSize);

    // Scale boxes back to original image dimensions
    final scaleX = image.width / _detSize;
    final scaleY = image.height / _detSize;

    return boxes.map((box) => _TextBox(box.x1 * scaleX, box.y1 * scaleY, box.x2 * scaleX, box.y2 * scaleY)).toList();
  }

  /// Resize image to fit within detSize×detSize, padding with zeros.
  img.Image _resizeForDetection(img.Image image) {
    final w = image.width;
    final h = image.height;
    final ratio = math.min(_detSize / w, _detSize / h);
    final newW = (w * ratio).round();
    final newH = (h * ratio).round();

    final resized = img.copyResize(image, width: newW, height: newH, interpolation: img.Interpolation.linear);

    // Create padded image (black background)
    final padded = img.Image(width: _detSize, height: _detSize);
    img.compositeImage(padded, resized, dstX: 0, dstY: 0);
    return padded;
  }

  /// Find bounding boxes from the probability map using a simplified
  /// connected-components approach (horizontal projection + vertical scan).
  List<_TextBox> _findBoxes(List<dynamic> probMap, int width, int height) {
    // Convert to binary mask
    final mask = List.generate(height, (y) {
      return List.generate(width, (x) {
        final idx = y * width + x;
        return idx < probMap.length && (probMap[idx] as num).toDouble() > _detThreshold;
      });
    });

    // Horizontal projection — find rows with text
    final rowHasText = List.generate(height, (y) {
      var count = 0;
      for (var x = 0; x < width; x++) {
        if (mask[y][x]) count++;
      }
      return count > 5; // at least 5 pixels in this row
    });

    // Find text line regions (vertical grouping)
    final boxes = <_TextBox>[];
    var inLine = false;
    var startY = 0;

    for (var y = 0; y <= height; y++) {
      final hasText = y < height && rowHasText[y];
      if (hasText && !inLine) {
        startY = y;
        inLine = true;
      } else if (!hasText && inLine) {
        // End of a text line — find horizontal extent
        var minX = width;
        var maxX = 0;
        for (var yy = startY; yy < y; yy++) {
          for (var xx = 0; xx < width; xx++) {
            if (mask[yy][xx]) {
              if (xx < minX) minX = xx;
              if (xx > maxX) maxX = xx;
            }
          }
        }
        if (maxX > minX) {
          boxes.add(_TextBox(minX.toDouble(), startY.toDouble(), maxX.toDouble(), (y - 1).toDouble()));
        }
        inLine = false;
      }
    }

    return boxes;
  }

  // ─── Recognition ───────────────────────────────────────────────────────

  /// Run the recognition model on a cropped text region.
  Future<String> _recognize(img.Image crop) async {
    // Resize to 48×320 (rec model input size)
    final resized = img.copyResize(crop, width: _recWidth, height: _recHeight, interpolation: img.Interpolation.linear);

    // Convert to NCHW float32, normalize to [-1, 1] for PP-OCR recognition
    final input = _imageToNchw(resized, _recWidth, _recHeight, toNegOne: true);

    // Run inference
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [1, 3, _recHeight, _recWidth]);
    final runOptions = OrtRunOptions();
    final outputs = await _recSession!.run(runOptions, {'x': inputTensor});
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty || outputs.first == null) return '';

    final outputValue = outputs.first!.value;
    outputs.first!.release();

    // Output is [1, 40, 438] — CTC decode
    final flat = _flattenToList(outputValue);
    if (flat.isEmpty) return '';

    return _ctcDecode(flat, 40, _dict!.length + 1);
  }

  /// CTC decode: take argmax per timestep, collapse repeats, remove blanks.
  /// PP-OCR uses blank at index 0. Characters start at index 1.
  String _ctcDecode(List<dynamic> output, int timesteps, int numClasses) {
    final chars = <int>[];
    var prevChar = -1;

    for (var t = 0; t < timesteps; t++) {
      var maxIdx = 0;
      var maxVal = -1.0;
      for (var c = 0; c < numClasses; c++) {
        final idx = t * numClasses + c;
        if (idx < output.length) {
          final val = (output[idx] as num).toDouble();
          if (val > maxVal) {
            maxVal = val;
            maxIdx = c;
          }
        }
      }

      // Skip blank (index 0) and repeats
      if (maxIdx != _blankIndex && maxIdx != prevChar) {
        // Map model output index to dict index: output 1 → dict[0], output 2 → dict[1], etc.
        chars.add(maxIdx - 1);
      }
      prevChar = maxIdx;
    }

    // Convert indices to characters
    final result = StringBuffer();
    for (final idx in chars) {
      if (idx >= 0 && idx < _dict!.length) {
        result.write(_dict![idx]);
      }
    }
    return result.toString();
  }

  // ─── Helpers ───────────────────────────────────────────────────────────

  /// Crop a text region from the image, expanding slightly for better recog.
  img.Image _cropBox(img.Image image, _TextBox box) {
    final x = box.x1.floor().clamp(0, image.width - 1);
    final y = box.y1.floor().clamp(0, image.height - 1);
    final w = (box.x2 - box.x1).ceil().clamp(1, image.width - x);
    final h = (box.y2 - box.y1).ceil().clamp(1, image.height - y);
    return img.copyCrop(image, x: x, y: y, width: w, height: h);
  }

  /// Convert image to NCHW Float32List for ONNX input.
  /// For detection: normalize to [0, 1] (divide by 255)
  /// For recognition: normalize to [-1, 1] ((x/255 - 0.5) / 0.5)
  Float32List _imageToNchw(img.Image image, int width, int height, {bool normalize = false, bool toNegOne = false}) {
    final pixelValues = Float32List(3 * height * width);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        double r, g, b;
        if (toNegOne) {
          // PP-OCR recognition: normalize to [-1, 1]
          r = (pixel.rNormalized.toDouble() - 0.5) / 0.5;
          g = (pixel.gNormalized.toDouble() - 0.5) / 0.5;
          b = (pixel.bNormalized.toDouble() - 0.5) / 0.5;
        } else if (normalize) {
          // Detection: normalize to [0, 1]
          r = pixel.rNormalized.toDouble();
          g = pixel.gNormalized.toDouble();
          b = pixel.bNormalized.toDouble();
        } else {
          r = pixel.r / 255.0;
          g = pixel.g / 255.0;
          b = pixel.b / 255.0;
        }
        final idx = y * width + x;
        pixelValues[0 * height * width + idx] = r;
        pixelValues[1 * height * width + idx] = g;
        pixelValues[2 * height * width + idx] = b;
      }
    }
    return pixelValues;
  }

  /// Recursively flatten nested lists.
  List<dynamic> _flattenToList(dynamic value) {
    final result = <dynamic>[];
    void recurse(dynamic v) {
      if (v is List) {
        for (final item in v) {
          recurse(item);
        }
      } else {
        result.add(v);
      }
    }

    recurse(value);
    return result;
  }
}

/// Bounding box for a text region.
class _TextBox {
  final double x1, y1, x2, y2;
  _TextBox(this.x1, this.y1, this.x2, this.y2);
}
