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
  final _controller = TextEditingController();
  Timer? _debounce;
  EffectCleanup? _externalSync;

  @override
  void initState() {
    super.initState();
    // Push external clears (e.g. Escape key) into the visible text.
    _externalSync = effect(() {
      final v = searchQuery.value;
      if (_controller.text != v) {
        _controller.value = TextEditingValue(
          text: v,
          selection: TextSelection.collapsed(offset: v.length),
        );
      }
    });
  }

  @override
  void dispose() {
    _debounce?.cancel();
    _externalSync?.call();
    _controller.dispose();
    super.dispose();
  }

  void _onChanged(String v) {
    searchQuery.value = v;
    _debounce?.cancel();
    _debounce = Timer(const Duration(milliseconds: 150), () {
      performSearch(v);
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.theme.colors;
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _controller,
            autofocus: true,
            style: TextStyle(color: colors.foreground),
            cursorColor: colors.primary,
            decoration: InputDecoration(
              hintText: 'Describe the reaction (e.g. "sarcastic fail")',
              hintStyle: TextStyle(color: colors.mutedForeground),
              prefixIcon: Icon(FLucideIcons.search, size: 16, color: colors.mutedForeground),
              suffixIcon: SignalBuilder(
                builder: (context) {
                  final hasText = searchQuery.value.isNotEmpty;
                  if (!hasText) return const SizedBox.shrink();
                  return IconButton(
                    iconSize: 16,
                    icon: Icon(FLucideIcons.x, color: colors.mutedForeground),
                    onPressed: () {
                      _controller.clear();
                      _onChanged('');
                    },
                  );
                },
              ),
              filled: true,
              fillColor: colors.muted,
              border: OutlineInputBorder(
                borderRadius: context.theme.style.borderRadius.md,
                borderSide: BorderSide(color: colors.border),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: context.theme.style.borderRadius.md,
                borderSide: BorderSide(color: colors.border),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: context.theme.style.borderRadius.md,
                borderSide: BorderSide(color: colors.primary, width: 1.5),
              ),
              contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              isDense: true,
            ),
            onChanged: _onChanged,
          ),
        ),
        const SizedBox(width: 8),
        FButton.icon(
          variant: FButtonVariant.ghost,
          onPress: () {
            // TODO: settings sheet
          },
          child: const Icon(FLucideIcons.settings),
        ),
      ],
    );
  }
}
