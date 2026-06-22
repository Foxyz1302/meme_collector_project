import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';

import 'gif_grid.dart';
import 'app_searchbar.dart';
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
        sidebar: SignalBuilder(
            builder: (context) => AppSidebar(selected: selectedNav.value)),
        child: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Column(
            spacing: 10,
            children: [
              MySearchBar(),
              Expanded(child: GifGrid()),
            ],
          ),
        ),
      ),
    );
  }
}
