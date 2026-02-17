import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import * as api from '@/lib/api';

interface AuthContextType {
    user: api.User | null;
    token: string | null;
    isLoading: boolean;
    login: (username: string, password: string) => Promise<void>;
    register: (username: string, password: string, orgId?: string) => Promise<void>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<api.User | null>(null);
    const [token, setToken] = useState<string | null>(() => localStorage.getItem('rag_token'));
    const [isLoading, setIsLoading] = useState(true);

    const fetchUser = useCallback(async () => {
        if (!token) {
            setIsLoading(false);
            return;
        }
        try {
            const u = await api.getMe();
            setUser(u);
        } catch {
            localStorage.removeItem('rag_token');
            setToken(null);
            setUser(null);
        } finally {
            setIsLoading(false);
        }
    }, [token]);

    useEffect(() => {
        fetchUser();
    }, [fetchUser]);

    const loginFn = async (username: string, password: string) => {
        const res = await api.login(username, password);
        localStorage.setItem('rag_token', res.access_token);
        setToken(res.access_token);
        const u = await api.getMe();
        setUser(u);
    };

    const registerFn = async (username: string, password: string, orgId = 'default') => {
        const res = await api.register(username, password, orgId);
        localStorage.setItem('rag_token', res.access_token);
        setToken(res.access_token);
        const u = await api.getMe();
        setUser(u);
    };

    const logout = () => {
        localStorage.removeItem('rag_token');
        setToken(null);
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, token, isLoading, login: loginFn, register: registerFn, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be used within AuthProvider');
    return ctx;
}
