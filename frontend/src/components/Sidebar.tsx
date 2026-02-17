import { useState, useEffect } from 'react';
import { useAuth } from '@/context/auth-context';
import { healthCheck } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import type { JSX } from 'react/jsx-runtime';

export type Page = 'chat' | 'documents' | 'settings';

interface SidebarProps {
    currentPage: Page;
    onNavigate: (page: Page) => void;
}

export default function Sidebar({ currentPage, onNavigate }: SidebarProps) {
    const { user, logout } = useAuth();
    const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

    useEffect(() => {
        const check = () => {
            healthCheck()
                .then(h => setIsHealthy(h.status === 'ok'))
                .catch(() => setIsHealthy(false));
        };
        check();
        const interval = setInterval(check, 30000);
        return () => clearInterval(interval);
    }, []);

    const navItems: { page: Page; label: string; icon: JSX.Element }[] = [
        {
            page: 'chat',
            label: 'Chat',
            icon: (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
                </svg>
            ),
        },
        {
            page: 'documents',
            label: 'Documents',
            icon: (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
            ),
        },
        {
            page: 'settings',
            label: 'Settings',
            icon: (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
            ),
        },
    ];

    return (
        <div className="w-64 h-full bg-zinc-900 border-r border-zinc-800 flex flex-col">
            {/* Logo */}
            <div className="p-4 flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center shrink-0">
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                    </svg>
                </div>
                <div>
                    <h1 className="text-sm font-bold text-zinc-100">RAG System</h1>
                    <div className="flex items-center gap-1.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${isHealthy === null ? 'bg-zinc-500' : isHealthy ? 'bg-emerald-400' : 'bg-red-400'
                            }`} />
                        <span className="text-[10px] text-zinc-500">
                            {isHealthy === null ? 'Checking...' : isHealthy ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>
                </div>
            </div>

            <Separator className="bg-zinc-800" />

            {/* Navigation */}
            <nav className="flex-1 p-3 space-y-1">
                {navItems.map(item => (
                    <button
                        key={item.page}
                        onClick={() => onNavigate(item.page)}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all cursor-pointer ${currentPage === item.page
                                ? 'bg-zinc-800 text-cyan-400'
                                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                            }`}
                    >
                        {item.icon}
                        {item.label}
                    </button>
                ))}
            </nav>

            <Separator className="bg-zinc-800" />

            {/* User section */}
            <div className="p-3">
                <div className="flex items-center gap-3 px-3 py-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500/30 to-violet-500/30 flex items-center justify-center">
                        <span className="text-xs font-bold text-cyan-300 uppercase">
                            {user?.username?.charAt(0) || '?'}
                        </span>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-zinc-200 truncate">{user?.username}</p>
                        <p className="text-[10px] text-zinc-500 capitalize">{user?.role}</p>
                    </div>
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={logout}
                        className="h-8 w-8 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 shrink-0 cursor-pointer"
                        title="Logout"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9" />
                        </svg>
                    </Button>
                </div>
            </div>
        </div>
    );
}
