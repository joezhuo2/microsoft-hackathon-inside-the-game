// Theme registry: the only place overlay accents, motifs and motion are defined.
window.THEMES = {
  breakthrough: { label: "Breakthrough", accent: "#22d3ee", icon: "breakthrough", motion: "sweep", priority: "medium" },
  pressure_shift: { label: "Pressure shift", accent: "#fb923c", icon: "pressure", motion: "jolt", priority: "medium" },
  momentum_swing: { label: "Momentum swing", accent: "#a78bfa", icon: "wave", motion: "swing", priority: "medium" },
  player_spotlight: { label: "Player spotlight", accent: "#facc15", icon: "spotlight", motion: "spot", priority: "medium" },
  goal_threat: { label: "Goal threat", accent: "#f87171", icon: "target", motion: "pulse", priority: "high" },
  defensive_masterclass: { label: "Defensive masterclass", accent: "#4ade80", icon: "shield", motion: "lock", priority: "medium" },
  chaos_index: { label: "Chaos index", accent: "#f472b6", icon: "chaos", motion: "flicker", priority: "low" },
  control_index: { label: "Control index", accent: "#60a5fa", icon: "control", motion: "settle", priority: "low" },
  value_swing: { label: "Value swing", accent: "#d4af37", icon: "coin", motion: "rise", priority: "medium" },
  milestone: { label: "Milestone", accent: "#e5e7eb", icon: "flag", motion: "rise", priority: "medium" },
  recap_cue: { label: "Recap", accent: "#cbd5e1", icon: "recap", motion: "settle", priority: "high" },
};

window.THEME_ICONS = {
  breakthrough: '<path d="M3 12h13M12 7l5 5-5 5"/><path d="M20 4v16" stroke-dasharray="2 3"/>',
  pressure: '<path d="M3 12h6M6 9l3 3-3 3M21 12h-6M18 9l-3 3 3 3"/><path d="M12 5v14"/>',
  spotlight: '<circle cx="12" cy="8" r="3.5"/><path d="M5 21c1-4 3.8-6 7-6s6 2 7 6"/><path d="M12 1.5v1.5M4.5 4.5l1 1M19.5 4.5l-1 1"/>',
  chaos: '<path d="M3 7h3.5c2 0 3 1 4.5 3.5S14 17 16.5 17H21M3 17h3.5c1.4 0 2.3-.5 3.2-1.6M14.3 8.6C15 7.5 15.9 7 17 7h4M18 4l3 3-3 3M18 14l3 3-3 3"/>',
  control: '<rect x="4" y="4" width="16" height="16"/><path d="M4 12h16M12 4v16"/>',
  coin: '<circle cx="12" cy="12" r="9"/><path d="M15 9.2c-.6-.8-1.6-1.2-3-1.2-1.7 0-2.8.8-2.8 2 0 3 6 1.4 6 4.2 0 1.2-1.2 2-3.2 2-1.4 0-2.5-.5-3.2-1.4M12 6v2M12 16v2"/>',
  flag: '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
  recap: '<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>',
};

window.theme = (id) => window.THEMES[id] || window.THEMES.momentum_swing;
