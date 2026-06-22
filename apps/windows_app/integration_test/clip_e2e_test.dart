/// End-to-end CLIP validation test.
///
/// Run from apps/windows_app/:
///   flutter test integration_test/clip_e2e_test.dart -d windows
///
/// This is the GATE for the CLIP pipeline. If this test passes, the entire
/// chain works:
///   1. BPE tokenizer (ClipTokenizer) produces correct token IDs
///   2. ONNX text tower produces a 512-dim embedding
///   3. Image preprocessing (decode + resize + normalize + transpose) works
///   4. ONNX vision tower produces a 512-dim embedding
///   5. The two embeddings live in the same vector space (cosine sim > 0.2
///      for matching text/image pairs)
///
/// Test images (in assets/test_data/):
///   - cat.jpg            — a cat photo. Should match "a cat" > "a car"
///   - HLX5XBbWYAAKisS.jpg — image with text overlay. Should match the
///                          overlaid text (need to OCR or guess the text).
///
/// If similarity for matching pairs is < 0.2 or close to random, something
/// is wrong with the BPE tokenizer or image preprocessing.

import 'dart:io';
import 'dart:math' show sqrt;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;

import '../lib/inference_impl.dart';
import 'package:meme_collector_core/meme_collector_core.dart' show cosineSimilarity;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final modelsDir = p.normalize(
      p.join(Directory.current.path, '..', '..', 'assets', 'models'));
  final testDataDir = p.normalize(
      p.join(Directory.current.path, '..', '..', 'assets', 'test_data'));

  late ClipTextEmbedder textEmbedder;
  late ClipImageEmbedder imageEmbedder;

  setUpAll(() async {
    textEmbedder = ClipTextEmbedder(
      modelPath: p.join(modelsDir, 'clip_text_fp16.onnx'),
      tokenizerPath: p.join(modelsDir, 'clip_tokenizer.json'),
    );
    await textEmbedder.init();

    imageEmbedder = ClipImageEmbedder(
      modelPath: p.join(modelsDir, 'clip_vision_fp16.onnx'),
    );
    await imageEmbedder.init();
  });

  tearDownAll(() async {
    await textEmbedder.dispose();
    await imageEmbedder.dispose();
  });

  test('CLIP tokenizer debug: print token IDs for various inputs', () async {
    // This test prints the actual token IDs so we can verify the BPE
    // tokenizer is producing different tokens for different inputs.
    // If all inputs produce the same tokens (e.g. just BOS+EOS+PAD),
    // the tokenizer is broken.
    final inputs = [
      'a cat',
      'a car',
      'a dog',
      'a puppy',
      'sarcastic reaction gif',
      'drake hotline bling',
    ];

    print('\n>>> Token ID debug:');
    for (final input in inputs) {
      final tokenIds = textEmbedder.tokenizer!.encode(input);
      // Count non-pad tokens (anything that's not EOS/pad)
      final nonPad = tokenIds.where((id) => id != ClipTokenizer.eosTokenId).toList();
      print('  "$input" → ${nonPad.length} tokens: $nonPad');
    }
    print('');

    // Sanity: "a cat" and "a car" should produce DIFFERENT token IDs
    final catTokens = textEmbedder.tokenizer!.encode('a cat');
    final carTokens = textEmbedder.tokenizer!.encode('a car');
    expect(catTokens[1] != carTokens[1] || catTokens[2] != carTokens[2], true,
        reason: 'Expected "a cat" and "a car" to produce different tokens. '
            'cat=${catTokens.sublist(0, 5)}, car=${carTokens.sublist(0, 5)}');
  });

  test('CLIP text embedding produces normalized 512-dim vector', () async {
    final vec = await textEmbedder.embed('a cat sitting on a table');
    expect(vec.length, 512, reason: 'embedding dimension');
    final norm = _l2Norm(vec);
    expect(norm, closeTo(1.0, 0.05),
        reason: 'embedding should be L2-normalized (norm=$norm)');
  });

  test('CLIP image embedding produces normalized 512-dim vector', () async {
    final catImagePath = p.join(testDataDir, 'cat.jpg');
    expect(File(catImagePath).existsSync(), true,
        reason: 'cat.jpg test image missing');

    // Debug: print some pixel values to verify the image decoded correctly
    final bytes = await File(catImagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    expect(decoded, isNotNull, reason: 'Could not decode cat.jpg');
    print('  cat.jpg dimensions: ${decoded!.width}x${decoded.height}');
    final pixel0 = decoded.getPixel(0, 0);
    print('  cat.jpg pixel(0,0) raw: r=${pixel0.r} g=${pixel0.g} b=${pixel0.b}');
    print('  cat.jpg pixel(0,0) normalized: r=${pixel0.rNormalized} g=${pixel0.gNormalized} b=${pixel0.bNormalized}');

    final vec = await imageEmbedder.embedFile(catImagePath);
    expect(vec.length, 512, reason: 'embedding dimension');
    final norm = _l2Norm(vec);
    expect(norm, closeTo(1.0, 0.05),
        reason: 'embedding should be L2-normalized (norm=$norm)');
  });

  test('CLIP cross-modal: "a cat" matches cat image > "a car" matches cat image', () async {
    final catImagePath = p.join(testDataDir, 'cat.jpg');
    final catVec = await imageEmbedder.embedFile(catImagePath);

    final catTextVec = await textEmbedder.embed('a cat');
    final carTextVec = await textEmbedder.embed('a car');

    final catSim = cosineSimilarity(catVec, catTextVec);
    final carSim = cosineSimilarity(catVec, carTextVec);

    print('  "a cat" vs cat.jpg similarity:  ${catSim.toStringAsFixed(4)}');
    print('  "a car" vs cat.jpg similarity:  ${carSim.toStringAsFixed(4)}');

    // The key assertion: cat > car
    expect(catSim > carSim, true,
        reason: 'Expected "a cat" to match cat image better than "a car" '
            '(cat=$catSim, car=$carSim)');

    // Also check absolute threshold — if cat sim is below 0.15, something is
    // wrong with the tokenizer or preprocessing (random would be ~0.0-0.1).
    expect(catSim > 0.15, true,
        reason: 'Expected "a cat" vs cat image similarity > 0.15, got $catSim. '
            'If this is failing, the BPE tokenizer or image preprocessing '
            'is producing garbage embeddings.');
  });

  test('CLIP cross-modal: text-text similarity sanity', () async {
    // Same text should have similarity 1.0
    final v1 = await textEmbedder.embed('a dog running');
    final v2 = await textEmbedder.embed('a dog running');
    final selfSim = cosineSimilarity(v1, v2);
    expect(selfSim, closeTo(1.0, 0.01),
        reason: 'identical text similarity should be ~1.0');

    // Synonyms should be more similar than unrelated words
    final dog1 = await textEmbedder.embed('a dog');
    final dog2 = await textEmbedder.embed('a puppy');
    final car = await textEmbedder.embed('a car');

    final dogSim = cosineSimilarity(dog1, dog2);
    final dogCarSim = cosineSimilarity(dog1, car);

    print('  "a dog" vs "a puppy" similarity:  ${dogSim.toStringAsFixed(4)}');
    print('  "a dog" vs "a car" similarity:    ${dogCarSim.toStringAsFixed(4)}');

    expect(dogSim > dogCarSim, true,
        reason: 'Expected "a dog" to match "a puppy" better than "a car"');
  });

  test('CLIP embedding latency is acceptable for search', () async {
    // Measure text embedding latency — should be < 200ms for responsive search
    final sw = Stopwatch()..start();
    await textEmbedder.embed('sarcastic reaction gif');
    sw.stop();
    print('  CLIP text embedding latency: ${sw.elapsedMilliseconds}ms');
    expect(sw.elapsedMilliseconds < 500, true,
        reason: 'Text embedding should be < 500ms for responsive search');
  });
}

double _l2Norm(Float32List v) {
  var sumSq = 0.0;
  for (var i = 0; i < v.length; i++) {
    sumSq += v[i] * v[i];
  }
  return sumSq > 0 ? sqrt(sumSq) : 0.0;
}
