import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';

import 'package:meme_collector_core/meme_collector_core.dart';

import 'gif_grid.dart';
import 'app_sidebar.dart';
import '../state/signals.dart';

class HomeScaffold extends StatefulWidget {
  const HomeScaffold({super.key});

  @override
  State<HomeScaffold> createState() => _HomeScaffoldState();
}

class _HomeScaffoldState extends State<HomeScaffold> {
  final _focus = FocusNode();

  @override
  void dispose() {
    _focus.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      focusNode: _focus,
      autofocus: true,
      onKeyEvent: (node, event) {
        if (event is KeyDownEvent) {
          // Ctrl+V — paste URL from clipboard
          if ((HardwareKeyboard.instance.isControlPressed || HardwareKeyboard.instance.isMetaPressed) &&
              event.logicalKey == LogicalKeyboardKey.keyV) {
            _pasteFromClipboard();
            return KeyEventResult.handled;
          }
          if (event.logicalKey == LogicalKeyboardKey.enter) {
            // TODO: copy first result
            return KeyEventResult.handled;
          }
          if (event.logicalKey == LogicalKeyboardKey.escape) {
            searchQuery.value = '';
            return KeyEventResult.handled;
          }
        }
        return KeyEventResult.ignored;
      },
      child: FScaffold(
        childPad: false,
        sidebar: SignalBuilder(builder: (context) => AppSidebar(selected: selectedNav.value)),
        child: const Material(color: Colors.transparent, child: GifGrid()),
      ),
    );
  }

  Future<void> _pasteFromClipboard() async {
    final clipData = await Clipboard.getData('text/plain');
    final url = clipData?.text?.trim();
    if (url == null || url.isEmpty) return;

    // Validate URL
    final validation = UrlValidator.validate(url);
    if (!validation.isValid) {
      showFToast(context: context, title: Text(validation.reason ?? 'Invalid URL'));
      return;
    }

    // Show toast that we're adding
    showFToast(context: context, title: const Text('Adding reaction…'));

    await addReaction(url);
  }
}
