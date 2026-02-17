import { useState } from 'react';
import { useAuth } from '@/context/auth-context';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function LoginPage() {
    const { login, register } = useAuth();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    // Login form
    const [loginUser, setLoginUser] = useState('');
    const [loginPass, setLoginPass] = useState('');

    // Register form
    const [regUser, setRegUser] = useState('');
    const [regPass, setRegPass] = useState('');
    const [regOrg, setRegOrg] = useState('default');

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');
        try {
            await login(loginUser, loginPass);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Login failed');
        } finally {
            setIsLoading(false);
        }
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');
        try {
            await register(regUser, regPass, regOrg);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Registration failed');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-zinc-950 via-zinc-900 to-zinc-950 p-4">
            {/* Decorative glows */}
            <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none" />
            <div className="fixed bottom-0 right-0 w-[400px] h-[400px] bg-violet-500/10 rounded-full blur-[120px] pointer-events-none" />

            <Card className="w-full max-w-md border-zinc-800/50 bg-zinc-900/80 backdrop-blur-xl shadow-2xl">
                <CardHeader className="text-center space-y-2">
                    <div className="mx-auto w-12 h-12 rounded-xl bg-linear-to-br from-cyan-500 to-violet-500 flex items-center justify-center mb-2">
                        <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
                        </svg>
                    </div>
                    <CardTitle className="text-2xl font-bold text-zinc-100">RAG System</CardTitle>
                    <CardDescription className="text-zinc-400">
                        Upload documents and query your knowledge base with AI
                    </CardDescription>
                </CardHeader>

                <CardContent>
                    {error && (
                        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                            {error}
                        </div>
                    )}

                    <Tabs defaultValue="login" className="w-full">
                        <TabsList className="grid w-full grid-cols-2 bg-zinc-800/50">
                            <TabsTrigger value="login" className="data-[state=active]:bg-zinc-700 cursor-pointer">Login</TabsTrigger>
                            <TabsTrigger value="register" className="data-[state=active]:bg-zinc-700 cursor-pointer">Register</TabsTrigger>
                        </TabsList>

                        <TabsContent value="login">
                            <form onSubmit={handleLogin} className="space-y-4 mt-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-zinc-300">Username</label>
                                    <Input
                                        value={loginUser}
                                        onChange={e => setLoginUser(e.target.value)}
                                        placeholder="Enter username"
                                        className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-zinc-300">Password</label>
                                    <Input
                                        type="password"
                                        value={loginPass}
                                        onChange={e => setLoginPass(e.target.value)}
                                        placeholder="Enter password"
                                        className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                                        required
                                    />
                                </div>
                                <Button
                                    type="submit"
                                    className="w-full bg-gradient-to-r from-cyan-600 to-violet-600 hover:from-cyan-500 hover:to-violet-500 text-white cursor-pointer"
                                    disabled={isLoading}
                                >
                                    {isLoading ? 'Signing in...' : 'Sign In'}
                                </Button>
                            </form>
                        </TabsContent>

                        <TabsContent value="register">
                            <form onSubmit={handleRegister} className="space-y-4 mt-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-zinc-300">Username</label>
                                    <Input
                                        value={regUser}
                                        onChange={e => setRegUser(e.target.value)}
                                        placeholder="Choose a username"
                                        className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                                        required
                                        minLength={3}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-zinc-300">Password</label>
                                    <Input
                                        type="password"
                                        value={regPass}
                                        onChange={e => setRegPass(e.target.value)}
                                        placeholder="Choose a password"
                                        className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                                        required
                                        minLength={6}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-zinc-300">Organization</label>
                                    <Input
                                        value={regOrg}
                                        onChange={e => setRegOrg(e.target.value)}
                                        placeholder="Organization ID"
                                        className="bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-500"
                                    />
                                </div>
                                <Button
                                    type="submit"
                                    className="w-full bg-gradient-to-r from-cyan-600 to-violet-600 hover:from-cyan-500 hover:to-violet-500 text-white cursor-pointer"
                                    disabled={isLoading}
                                >
                                    {isLoading ? 'Creating account...' : 'Create Account'}
                                </Button>
                            </form>
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </div>
    );
}
