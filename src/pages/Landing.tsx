import { Link } from 'react-router-dom'
import { LineChart, Compass, Search, MessageCircleHeart, Sprout, HeartHandshake } from 'lucide-react'
import { Logo } from '../components/Logo'

const features = [
  [LineChart, 'AI risk monitoring', 'A screening indicator that tracks change, not a verdict.'],
  [Compass, 'Behavioral intelligence', 'Sleep and activity compared with your own baseline.'],
  [Search, 'Explainable AI', 'SHAP shows which factors shaped each estimate.'],
  [Sprout, 'Personal wellness insights', 'Gentle, non-judgmental suggestions you can act on.'],
  [MessageCircleHeart, 'AI companion', 'A listen-first space that encourages real human contact.'],
  [HeartHandshake, 'Human-centered support', 'High risk is routed to people and helplines.'],
] as const

export default function Landing() {
  return (
    <div>
      <header className="sticky top-0 z-10 border-b border-line bg-cream/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-5 py-2">
          <Logo className="h-12" />
          <div className="ml-auto flex gap-2">
            <Link to="/login" className="flex min-h-11 items-center rounded-xl border border-forest px-5 font-semibold text-forest">Log in</Link>
            <Link to="/signup" className="flex min-h-11 items-center rounded-xl bg-forest px-5 font-semibold text-white">Get started</Link>
          </div>
        </div>
      </header>
      <section className="mx-auto max-w-6xl px-5 py-16">
        <span className="rounded-full bg-lav px-4 py-1 text-sm font-semibold">Research prototype · not a diagnosis</span>
        <h1 className="mt-4 max-w-3xl text-4xl font-extrabold leading-tight tracking-tight md:text-6xl">Your mental wellness, <span className="text-teal">understood</span> through AI.</h1>
        <p className="mt-5 max-w-xl text-lg text-mute">Discover behavioral patterns, understand your wellness trends, and access personalized support through intelligent, privacy-conscious technology.</p>
        <Link to="/signup" className="mt-6 inline-flex min-h-12 items-center rounded-xl bg-forest px-6 font-semibold text-white">Explore MindCare</Link>
      </section>
      <section className="mx-auto grid max-w-6xl gap-4 px-5 pb-16 sm:grid-cols-2 lg:grid-cols-3">
        {features.map(([Icon, t, d]) => (
          <div key={t} className="rounded-3xl border border-line bg-white p-6 shadow-sm">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-mint"><Icon size={20} aria-hidden /></div>
            <h3 className="mt-3 mb-1 text-lg font-bold">{t}</h3><p className="m-0 text-mute">{d}</p>
          </div>
        ))}
      </section>
      <section className="mx-auto max-w-6xl px-5 pb-16">
        <h2 className="text-3xl font-extrabold">Built on public research data</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {[['PSYCHE-D', 'Fitbit steps and sleep with PHQ-9 scores. The primary dataset.'], ['StudentLife', 'Smartphone sensing for about 49 students. Descriptive analysis only.'], ['WESAD', 'Wrist and chest physiology for stress and affect. A separate experiment.']].map(([t, d]) => (
            <div key={t} className="rounded-3xl bg-lav p-6"><h3 className="m-0 mb-1 text-lg font-bold">{t}</h3><p className="m-0">{d}</p></div>
          ))}
        </div>
      </section>
      <footer className="border-t border-line px-5 py-8 text-center text-sm text-mute">MindCare AI is a research prototype. Risk indicators are not diagnoses. In India, free 24x7 help: Tele-MANAS 14416. Emergency: 112.</footer>
    </div>
  )
}
