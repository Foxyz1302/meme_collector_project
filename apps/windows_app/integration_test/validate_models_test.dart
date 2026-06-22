/// ONNX + model2vec validation script.
///
/// Run from apps/windows_app/:
///   flutter test integration_test/validate_models_test.dart -d windows
///
/// This is the GATE before writing any real inference code. If any model
/// fails to load or produces garbage output, fix or find an alternative
/// before proceeding.
///
/// Uses onnxruntime_v2 (not flutter_onnxruntime) for GPU support:
///   - DirectML on Windows (GTX 1080, 3080 Ti, etc.)
///   - CUDA on Linux (future server)
///   - appendDefaultProviders() auto-selects best available EP with CPU fallback
///
/// What it validates:
///   1. Available execution providers (DirectML? CUDA? CPU only?)
///   2. CLIP text tower (text_model_fp16.onnx) — loads, accepts synthetic input
///   3. CLIP vision tower (vision_model_fp16.onnx) — loads, accepts synthetic input
///   4. PP-OCRv5 detection (det.onnx) — loads, accepts synthetic input
///   5. PP-OCRv5 recognition (rec.onnx) — loads, accepts synthetic input
///   6. model2vec (potion-base-32M) — file presence check (real API wiring TODO)

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:path/path.dart' as p;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Models live at workspace_root/assets/models/. From apps/windows_app/,
  // that's ../../assets/models/.
  final modelsDir = p.normalize(
      p.join(Directory.current.path, '..', '..', 'assets', 'models'));
  final report = <String, ModelReport>{};

  print('Models dir: $modelsDir');

  setUpAll(() {
    // Initialize ORT environment once.
    OrtEnv.instance.init();
  });

  test('detect available execution providers', () async {
    final providers = OrtEnv.instance.availableProviders();
    // OrtProvider is an enum — convert to strings for printing
    final providerNames = providers.map((p) => p.toString()).toList();
    print('\n>>> Available ORT execution providers: $providerNames');
    print('>>> Provider count: ${providers.length}');
    for (var i = 0; i < providers.length; i++) {
      print('>>>   [$i] ${providers[i]} (index=${providers[i].index})');
    }
    report['execution_providers'] = ModelReport(
      loaded: true,
      sanityPassed: true,
      inputNames: providerNames,
    );
    // Just print, don't assert — we want to know what's available even if GPU EPs aren't.
  });

  test('validate CLIP text tower (text_model_fp16.onnx) on CPU', () async {
    final result = await _validateOnnxModel(
      name: 'clip_text_cpu',
      modelPath: p.join(modelsDir, 'clip_text_fp16.onnx'),
      syntheticInputs: _syntheticClipTextInputs(),
      useGpu: false,
      outputLen: 512,
    );
    report['clip_text_cpu'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'CLIP text failed to load on CPU');
  });

  test('validate CLIP text tower (text_model_fp16.onnx) on GPU', () async {
    final result = await _validateOnnxModel(
      name: 'clip_text_gpu',
      modelPath: p.join(modelsDir, 'clip_text_fp16.onnx'),
      syntheticInputs: _syntheticClipTextInputs(),
      useGpu: true,
      outputLen: 512,
    );
    report['clip_text_gpu'] = result;
    // Don't fail if GPU unavailable — just record.
    if (!result.loaded) {
      print('  [clip_text_gpu] SKIPPED: ${result.error}');
    }
  });

  test('validate CLIP vision tower (vision_model_fp16.onnx) on CPU', () async {
    final result = await _validateOnnxModel(
      name: 'clip_vision_cpu',
      modelPath: p.join(modelsDir, 'clip_vision_fp16.onnx'),
      syntheticInputs: _syntheticClipVisionInputs(),
      useGpu: false,
      outputLen: 512,
    );
    report['clip_vision_cpu'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'CLIP vision failed to load on CPU');
  });

  test('validate CLIP vision tower (vision_model_fp16.onnx) on GPU', () async {
    final result = await _validateOnnxModel(
      name: 'clip_vision_gpu',
      modelPath: p.join(modelsDir, 'clip_vision_fp16.onnx'),
      syntheticInputs: _syntheticClipVisionInputs(),
      useGpu: true,
      outputLen: 512,
    );
    report['clip_vision_gpu'] = result;
    if (!result.loaded) {
      print('  [clip_vision_gpu] SKIPPED: ${result.error}');
    }
  });

  test('validate PP-OCRv5 detection (det.onnx) on CPU', () async {
    final result = await _validateOnnxModel(
      name: 'ppocr_det_cpu',
      modelPath: p.join(modelsDir, 'ppocr_det.onnx'),
      syntheticInputs: _syntheticPpocrDetInputs(),
      useGpu: false,
      outputLen: null,
    );
    report['ppocr_det_cpu'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'PP-OCR det failed to load on CPU');
  });

  test('validate PP-OCRv5 recognition (rec.onnx) on CPU', () async {
    final result = await _validateOnnxModel(
      name: 'ppocr_rec_cpu',
      modelPath: p.join(modelsDir, 'ppocr_rec.onnx'),
      syntheticInputs: _syntheticPpocrRecInputs(),
      useGpu: false,
      outputLen: null,
    );
    report['ppocr_rec_cpu'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'PP-OCR rec failed to load on CPU');
  });

  test('validate model2vec (potion-base-32M) file presence', () async {
    final result = _validateModel2VecFiles(p.join(modelsDir, 'potion-base-32M'));
    report['model2vec'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'model2vec files missing');
  });

  tearDownAll(() async {
    // Write the report to disk for inspection
    final reportPath = p.normalize(p.join(
        Directory.current.path, '..', '..', 'scripts', 'validation_report.json'));
    final reportFile = File(reportPath);
    await reportFile.parent.create(recursive: true);
    final reportJson = report.map((k, v) => MapEntry(k, v.toJson()));
    await reportFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(reportJson),
      flush: true,
    );

    print('\n${'=' * 60}');
    print('VALIDATION REPORT');
    print('${'=' * 60}');
    for (final entry in report.entries) {
      final r = entry.value;
      print('\n${entry.key}:');
      print('  loaded:        ${r.loaded}');
      print('  load_ms:       ${r.loadMs?.toStringAsFixed(1) ?? 'N/A'}');
      print('  infer_ms:      ${r.inferMs?.toStringAsFixed(1) ?? 'N/A'}');
      print('  output_len:    ${r.outputShape?.firstOrNull ?? 'N/A'}');
      print('  sanity_passed: ${r.sanityPassed}');
      if (r.error != null) print('  error:         ${r.error}');
      if (r.inputNames != null) print('  input_names:   ${r.inputNames}');
      if (r.outputNames != null) print('  output_names:  ${r.outputNames}');
    }
    print('\nReport saved to: $reportPath');

    // Print GPU vs CPU summary if we have both
    final cpuText = report['clip_text_cpu'];
    final gpuText = report['clip_text_gpu'];
    final cpuVision = report['clip_vision_cpu'];
    final gpuVision = report['clip_vision_gpu'];
    print('\n${'-' * 60}');
    print('GPU vs CPU SUMMARY');
    print('${'-' * 60}');
    if (cpuText != null && gpuText != null && gpuText.loaded) {
      final speedup = (cpuText.inferMs ?? 1) / (gpuText.inferMs ?? 1);
      print('CLIP text:  CPU ${cpuText.inferMs?.toStringAsFixed(1)}ms  |  '
          'GPU ${gpuText.inferMs?.toStringAsFixed(1)}ms  |  '
          'speedup ${speedup.toStringAsFixed(2)}×');
    } else if (gpuText != null && !gpuText.loaded) {
      print('CLIP text:  GPU unavailable (${gpuText.error})');
    }
    if (cpuVision != null && gpuVision != null && gpuVision.loaded) {
      final speedup = (cpuVision.inferMs ?? 1) / (gpuVision.inferMs ?? 1);
      print('CLIP vision: CPU ${cpuVision.inferMs?.toStringAsFixed(1)}ms  |  '
          'GPU ${gpuVision.inferMs?.toStringAsFixed(1)}ms  |  '
          'speedup ${speedup.toStringAsFixed(2)}×');
    } else if (gpuVision != null && !gpuVision.loaded) {
      print('CLIP vision: GPU unavailable (${gpuVision.error})');
    }
  });
}

// ─── Report data class ──────────────────────────────────────────────────────

class ModelReport {
  final bool loaded;
  final double? loadMs;
  final double? inferMs;
  final List<int>? outputShape; // we store output length here (single int in list)
  final List<String>? inputNames;
  final List<String>? outputNames;
  final bool sanityPassed;
  final String? error;

  ModelReport({
    required this.loaded,
    this.loadMs,
    this.inferMs,
    this.outputShape,
    this.inputNames,
    this.outputNames,
    required this.sanityPassed,
    this.error,
  });

  Map<String, dynamic> toJson() => {
        'loaded': loaded,
        if (loadMs != null) 'load_ms': loadMs,
        if (inferMs != null) 'infer_ms': inferMs,
        if (outputShape != null) 'output_shape': outputShape,
        if (inputNames != null) 'input_names': inputNames,
        if (outputNames != null) 'output_names': outputNames,
        'sanity_passed': sanityPassed,
        if (error != null) 'error': error,
      };
}

// ─── ONNX model validation ──────────────────────────────────────────────────

class _SyntheticInput {
  final String name;
  final List<int> shape;
  final List<dynamic> data;

  _SyntheticInput({
    required this.name,
    required this.shape,
    required this.data,
  });
}

Future<ModelReport> _validateOnnxModel({
  required String name,
  required String modelPath,
  required List<_SyntheticInput> syntheticInputs,
  required bool useGpu,
  int? outputLen, // expected length of the output (flattened)
}) async {
  final sw = Stopwatch()..start();
  try {
    final modelFile = File(modelPath);
    if (!await modelFile.exists()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'Model file not found: $modelPath',
      );
    }

    // Read model bytes from file (OrtSession.fromBuffer takes bytes)
    final modelBytes = await modelFile.readAsBytes();

    // Configure session options with EP selection
    final sessionOptions = OrtSessionOptions();
    if (useGpu) {
      // appendDefaultProviders() didn't pick up DirectML in the previous run
      // (showed [CPU, CPU] instead). Try DirectML explicitly first, then fall
      // back to appendDefaultProviders() if the method doesn't exist or fails.
      var directMlLoaded = false;
      try {
        // ignore: avoid_dynamic_calls
        (sessionOptions as dynamic).appendDirectMLProvider();
        directMlLoaded = true;
        print('  [$name] DirectML provider appended explicitly');
      } catch (e) {
        print('  [$name] appendDirectMLProvider failed: $e');
      }
      if (!directMlLoaded) {
        try {
          sessionOptions.appendDefaultProviders();
          print('  [$name] appendDefaultProviders() called (fallback)');
        } catch (e) {
          print('  [$name] appendDefaultProviders failed: $e');
        }
      }
    }
    // For CPU-only: don't append any providers, ORT defaults to CPU.

    final session = OrtSession.fromBuffer(modelBytes, sessionOptions);

    sw.stop();
    final loadMs = sw.elapsedMilliseconds.toDouble();

    // Get input/output names (sync properties on OrtSession)
    final inputNames = session.inputNames;
    final outputNames = session.outputNames;

    print('  [$name] inputs:  $inputNames');
    print('  [$name] outputs: $outputNames');

    // Try inference with synthetic input
    sw..reset()..start();

    final inputsMap = <String, OrtValue>{};
    final inputsToRelease = <OrtValue>[];
    for (final synth in syntheticInputs) {
      // OrtValueTensor.createTensorWithDataList(data, [shape])
      final tensor =
          OrtValueTensor.createTensorWithDataList(synth.data, synth.shape);
      inputsMap[synth.name] = tensor;
      inputsToRelease.add(tensor);
    }

    final runOptions = OrtRunOptions();
    // runAsync returns Future<List<OrtValue?>?> — positional list, not a Map
    final outputs = await session.runAsync(runOptions, inputsMap);

    sw.stop();
    final inferMs = sw.elapsedMilliseconds.toDouble();

    // Release inputs + runOptions
    for (final t in inputsToRelease) {
      t.release();
    }
    runOptions.release();

    // Read output values
    int? actualOutputLen;
    if (outputs != null && outputs.isNotEmpty) {
      final firstOutput = outputs.first;
      if (firstOutput != null) {
        try {
          // OrtValueTensor.value returns dynamic. For shape [1, 512] it
          // returns a nested List<List<double>> — outer length 1, inner
          // length 512. We need to flatten.
          final value = firstOutput.value;
          print('  [$name] output runtimeType: ${value.runtimeType}');

          // Recursively flatten nested lists to a flat List<double>
          final flattened = _flattenToList(value);
          actualOutputLen = flattened.length;
          print('  [$name] output flattened len: ${flattened.length}');
          if (flattened.isNotEmpty) {
            final preview = flattened.take(5).toList();
            print('  [$name] output[0..5]: $preview');
          }

          // Also print the raw structure for debugging
          if (value is List && value.length <= 5) {
            print('  [$name] output raw (small): $value');
          }
        } catch (e, st) {
          print('  [$name] could not read output: $e');
          print('  [$name] stack: $st');
        }
        // Release output tensor
        firstOutput.release();
      }
      // Release any remaining outputs
      for (var i = 1; i < outputs.length; i++) {
        outputs[i]?.release();
      }
    }

    session.release();

    // Verify output length if expected
    var shapeOk = true;
    if (outputLen != null && actualOutputLen != null) {
      shapeOk = actualOutputLen == outputLen;
    }

    return ModelReport(
      loaded: true,
      loadMs: loadMs,
      inferMs: inferMs,
      outputShape: actualOutputLen != null ? [actualOutputLen] : null,
      inputNames: inputNames,
      outputNames: outputNames,
      sanityPassed: shapeOk,
      error: shapeOk
          ? null
          : 'Output len $actualOutputLen != expected $outputLen',
    );
  } catch (e, st) {
    sw.stop();
    print('  [$name] EXCEPTION: $e');
    print('  [$name] STACK: $st');
    return ModelReport(
      loaded: false,
      sanityPassed: false,
      error: e.toString(),
    );
  }
}

// ─── Synthetic input generators ─────────────────────────────────────────────

/// CLIP text tower expects only one input: input_ids (int32[1, 77]).
/// No attention_mask — the text tower uses causal attention internally.
List<_SyntheticInput> _syntheticClipTextInputs() {
  final seqLen = 77;
  final inputIds = List<int>.filled(seqLen, 0);
  inputIds[0] = 49406; // BOS
  inputIds[1] = 320; // "a"
  inputIds[2] = 2368; // "cat"
  inputIds[seqLen - 1] = 49407; // EOS

  return [
    _SyntheticInput(name: 'input_ids', shape: [1, seqLen], data: inputIds),
  ];
}

/// CLIP vision tower expects: pixel_values (float32[1, 3, 224, 224]).
List<_SyntheticInput> _syntheticClipVisionInputs() {
  final width = 224, height = 224, channels = 3;
  final total = channels * height * width;
  final rng = math.Random(42);
  final pixelValues = Float32List(total);
  for (var i = 0; i < total; i++) {
    pixelValues[i] = (rng.nextDouble() * 2 - 1).toDouble(); // [-1, 1] range
  }
  return [
    _SyntheticInput(
        name: 'pixel_values',
        shape: [1, channels, height, width],
        data: pixelValues),
  ];
}

/// PP-OCRv5 detection model input: x (float32[1, 3, 960, 960]).
List<_SyntheticInput> _syntheticPpocrDetInputs() {
  final size = 960;
  final total = 3 * size * size;
  final rng = math.Random(42);
  final pixelValues = Float32List(total);
  for (var i = 0; i < total; i++) {
    pixelValues[i] = rng.nextDouble().toDouble();
  }
  return [
    _SyntheticInput(name: 'x', shape: [1, 3, size, size], data: pixelValues),
  ];
}

/// PP-OCRv5 recognition model input: x (float32[1, 3, 48, 320]).
List<_SyntheticInput> _syntheticPpocrRecInputs() {
  final h = 48, w = 320;
  final total = 3 * h * w;
  final rng = math.Random(42);
  final pixelValues = Float32List(total);
  for (var i = 0; i < total; i++) {
    pixelValues[i] = rng.nextDouble().toDouble();
  }
  return [
    _SyntheticInput(name: 'x', shape: [1, 3, h, w], data: pixelValues),
  ];
}

// ─── model2vec file presence check ──────────────────────────────────────────

ModelReport _validateModel2VecFiles(String modelDir) {
  try {
    final dir = Directory(modelDir);
    if (!dir.existsSync()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'Model dir not found: $modelDir',
      );
    }

    final modelFile = File(p.join(modelDir, 'model.safetensors'));
    final tokenizerFile = File(p.join(modelDir, 'tokenizer.json'));
    if (!modelFile.existsSync()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'model.safetensors not found',
      );
    }
    if (!tokenizerFile.existsSync()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'tokenizer.json not found',
      );
    }

    final modelSize = modelFile.lengthSync();
    if (modelSize < 50 * 1024 * 1024) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'model.safetensors too small: $modelSize bytes',
      );
    }

    return ModelReport(
      loaded: true,
      sanityPassed: true,
      inputNames: const ['(files only — real API wiring in next step)'],
    );
  } catch (e) {
    return ModelReport(
      loaded: false,
      sanityPassed: false,
      error: e.toString(),
    );
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Recursively flatten a nested List structure into a flat List.
///
/// OrtValueTensor.value returns data shaped like the output tensor:
///   - shape [1, 512] → List<List<double>> with outer len 1, inner len 512
///   - shape [1, 1, 960, 960] → 4-level nested
///   - shape [1, 40, 438] → 3-level nested
///
/// For validation we just want the total element count, so flatten everything.
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
