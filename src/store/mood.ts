import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface MoodEntry { v: number; note: string; date: string }
export const useMood = create<{ entries: MoodEntry[]; add: (e: MoodEntry) => void; clear: () => void }>()(
  persist((set) => ({ entries: [], add: (e) => set((s) => ({ entries: [...s.entries, e] })), clear: () => set({ entries: [] }) }), { name: 'mindcare-mood' }),
)
