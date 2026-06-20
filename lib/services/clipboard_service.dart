// lib/services/clipboard_service.dart
import 'package:flutter/services.dart';
import '../models/gif_media.dart';

class ClipboardService {
  /// TODO: copy direct URL if from tenor/giphy, else copy file path / upload.
  static Future<void> copyReaction(GifMedia item) async {
    await Clipboard.setData(ClipboardData(text: item.sourceUrl ?? item.filePath));
  }
}
