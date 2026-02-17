import { useState } from 'react';
import { AuthProvider, useAuth } from '@/context/auth-context';
import LoginPage from '@/pages/LoginPage';
import ChatPage from '@/pages/ChatPage';
import DocumentsPage from '@/pages/DocumentsPage';
import SettingsPage from '@/pages/SettingsPage';
import Sidebar, { type Page } from '@/components/Sidebar';

function AppContent() {
  const { user, isLoading } = useAuth();
  const [page, setPage] = useState<Page>('chat');

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-zinc-950">
        <div className="flex flex-col items-center gap-3">
          <svg className="w-8 h-8 text-cyan-400 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className="text-sm text-zinc-500">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <LoginPage />;
  }

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100">
      <Sidebar currentPage={page} onNavigate={setPage} />
      <main className="flex-1 overflow-hidden">
        {page === 'chat' && <ChatPage />}
        {page === 'documents' && <DocumentsPage />}
        {page === 'settings' && <SettingsPage />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
