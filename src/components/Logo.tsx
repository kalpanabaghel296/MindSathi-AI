// Replace the files in /public/logo with your own logo to rebrand.
export function Logo({ variant = 'full', className = 'h-10' }: { variant?: 'full' | 'icon'; className?: string }) {
  return <img src={variant === 'icon' ? '/logo/mindcare-icon.png' : '/logo/mindcare-logo.png'} alt="MindCare AI" className={className} />
}
