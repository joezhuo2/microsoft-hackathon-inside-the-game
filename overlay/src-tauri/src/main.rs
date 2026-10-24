#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tauri::{Emitter, Manager, State, WebviewWindow};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

static LOCKED: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct HitRegions(Mutex<Vec<[f64; 4]>>);

#[tauri::command]
fn set_hit_regions(regions: State<HitRegions>, rects: Vec<[f64; 4]>) {
    *regions.0.lock().unwrap() = rects;
}

fn cursor_over_controls(window: &WebviewWindow) -> Option<bool> {
    let cursor = window.cursor_position().ok()?;
    let origin = window.inner_position().ok()?;
    let scale = window.scale_factor().ok()?;
    let x = (cursor.x - origin.x as f64) / scale;
    let y = (cursor.y - origin.y as f64) / scale;
    let regions = window.state::<HitRegions>();
    let rects = regions.0.lock().ok()?;
    Some(rects.iter().any(|[left, top, width, height]| x >= *left && x <= left + width && y >= *top && y <= top + height))
}

fn watch_cursor(window: WebviewWindow) {
    thread::spawn(move || {
        let mut ignoring: Option<bool> = None;
        loop {
            thread::sleep(Duration::from_millis(30));
            let ignore = LOCKED.load(Ordering::SeqCst) || !cursor_over_controls(&window).unwrap_or(false);
            if ignoring != Some(ignore) && window.set_ignore_cursor_events(ignore).is_ok() {
                ignoring = Some(ignore);
            }
        }
    });
}

fn main() {
    let toggle = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyO);
    tauri::Builder::default()
        .manage(HitRegions::default())
        .invoke_handler(tauri::generate_handler![set_hit_regions])
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if shortcut == &toggle && event.state() == ShortcutState::Pressed {
                        let locked = !LOCKED.fetch_xor(true, Ordering::SeqCst);
                        let _ = app.emit("click-through", locked);
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(toggle)?;
            let window = app.get_webview_window("main").expect("main window");
            window.set_ignore_cursor_events(true)?;
            watch_cursor(window);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running match overlay");
}
