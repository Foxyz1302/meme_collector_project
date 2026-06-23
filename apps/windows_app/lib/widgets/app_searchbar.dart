import 'dart:async';
import 'package:flutter/material.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../state/signals.dart';

class MySearchBar extends StatefulWidget {
  const MySearchBar({super.key});

  @override
  State<MySearchBar> createState() => _MySearchBarState();
}

class _MySearchBarState extends State<MySearchBar> {
  Timer? _debounce;

  @override
  void dispose() {
    _debounce?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(4),
      child: FCard.raw(
        child: Padding(
          padding: const EdgeInsets.all(4),
          child: Row(
            spacing: 4,
            children: [
              const Icon(FLucideIcons.search, size: 16),
              Expanded(
                child: FTextField(
                  autofocus: true,
                  // Lifted control: text is driven directly by searchQuery
                  control: .managed(
                    initial: TextEditingValue(text: searchQuery.value),
                    onChange: (value) {
                      searchQuery.value = value.text;
                      _debounce?.cancel();
                      _debounce = Timer(const Duration(milliseconds: 150), () {
                        performSearch(value.text);
                      });
                    },
                  ),
                  // Built‑in clear button (appears when the field is not empty)
                  clearable: (value) => value.text.isNotEmpty,
                  hint: 'Describe the reaction (e.g. "sarcastic fail")',
                ),
              ),
              FButton.icon(
                variant: FButtonVariant.ghost,
                onPress: () {
                  // TODO: settings sheet
                },
                child: const Icon(FLucideIcons.settings),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
