import { useEffect, useState } from 'react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { analyticsService, riskService } from '../services/api'
import { THRESHOLDS } from '../data/mock'

export default function Overview() {
  const [risk, setRisk] = useState<number[]>([])
  const [sleep, setSleep] = useState<{ day: string; hours: number }[]>([])
  useEffect(() => { riskService.history().then(setRisk); analyticsService.sleep().then(setSleep) }, [])
  const p = risk.at(-1) ?? 0
  const level = p >= THRESHOLDS.high ? 'High' : p >= THRESHOLDS.moderate ? 'Moderate' : 'Low'
  return (
    <div>
      <h1 className="m-0 text-3xl font-extrabold">Welcome back!</h1>
      <p className="mt-1 mb-6 text-mute">Here is an overview of your behavioral patterns and wellness trends.</p>
      <div className="grid gap-4 lg:grid-cols-3">
        <section className="rounded-3xl bg-gradient-to-br from-forest to-teal p-6 text-white">
          <p className="m-0 text-xs font-bold tracking-widest uppercase opacity-80">Risk indicator</p>
          <p className="my-2 text-5xl font-extrabold">{Math.round(p * 100)}%</p>
          <p className="m-0 text-xl font-bold">{level}</p>
          <p className="mt-3 text-sm opacity-90">Illustrative demo value. A research estimate, not a diagnosis. Level cut-offs (27% and 42%) come from the project model.</p>
        </section>
        <section className="rounded-3xl border border-line bg-white p-6 lg:col-span-2">
          <p className="m-0 text-xs font-bold tracking-widest text-mute uppercase">Risk history (8 weeks, demo)</p>
          <div className="h-48"><ResponsiveContainer><AreaChart data={risk.map((v, i) => ({ w: `W${i + 1}`, v: Math.round(v * 100) }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E4DDCF" /><XAxis dataKey="w" /><YAxis domain={[0, 60]} /><Tooltip />
            <Area dataKey="v" stroke="#2A6F6B" fill="#CFE5DA" strokeWidth={3} name="Risk %" /></AreaChart></ResponsiveContainer></div>
        </section>
        <section className="rounded-3xl border border-line bg-white p-6 lg:col-span-3">
          <p className="m-0 text-xs font-bold tracking-widest text-mute uppercase">Sleep duration (hours, demo)</p>
          <div className="h-48"><ResponsiveContainer><BarChart data={sleep}><CartesianGrid strokeDasharray="3 3" stroke="#E4DDCF" /><XAxis dataKey="day" /><YAxis /><Tooltip /><Bar dataKey="hours" fill="#2A6F6B" radius={[8, 8, 0, 0]} /></BarChart></ResponsiveContainer></div>
        </section>
      </div>
    </div>
  )
}
