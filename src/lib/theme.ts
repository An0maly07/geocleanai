import { useSyncExternalStore } from 'react'

export type Theme = 'light' | 'dark'
const KEY = 'geoclean-theme', EVENT = 'geoclean-theme-change'
const query = () => window.matchMedia('(prefers-color-scheme: dark)')

function saved(): Theme | null {
  try { const value = localStorage.getItem(KEY); return value === 'light' || value === 'dark' ? value : null } catch { return null }
}
// index.html sets data-theme before first paint with the same rule: saved choice, else the system setting.
export function currentTheme(): Theme { return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light' }
function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#0e1411' : '#f7f8f4')
  window.dispatchEvent(new Event(EVENT))
}
export function setTheme(theme: Theme) {
  try { localStorage.setItem(KEY, theme) } catch { /* private mode: the choice lasts for this visit */ }
  apply(theme)
}
function subscribe(onChange: () => void) {
  const system = () => { if (!saved()) apply(query().matches ? 'dark' : 'light') }
  window.addEventListener(EVENT, onChange); query().addEventListener('change', system)
  return () => { window.removeEventListener(EVENT, onChange); query().removeEventListener('change', system) }
}
export function useTheme(): Theme { return useSyncExternalStore(subscribe, currentTheme, () => 'light') }
