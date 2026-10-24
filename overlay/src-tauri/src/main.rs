#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicBool, Ordering};

use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

static CLICK_THROUGH: AtomicBool = AtomicBool::new(false);

fn main() {
    let toggle = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyO);
    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if shortcut == &toggle && event.state() == ShortcutState::Pressed {
                        let on = !CLICK_THROUGH.fetch_xor(true, Ordering::SeqCst);
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.set_ignore_cursor_events(on);
                            let _ = window.emit("click-through", on);
                        }
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(toggle)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running match overlay");
}
