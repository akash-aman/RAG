import { useState, useEffect } from 'react';
import { useAuth } from '@/context/auth-context';
import { getUsers, healthCheck, type User, type HealthResponse } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

export default function SettingsPage() {
    const { user } = useAuth();
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [users, setUsers] = useState<User[]>([]);
    const [usersError, setUsersError] = useState('');

    const isAdmin = user?.role === 'admin';

    useEffect(() => {
        healthCheck().then(setHealth).catch(() => { });
        if (isAdmin) {
            getUsers().then(setUsers).catch(err => {
                setUsersError(err instanceof Error ? err.message : 'Failed to load users');
            });
        }
    }, [isAdmin]);

    return (
        <div className="flex flex-col h-full p-6 space-y-6 overflow-y-auto">
            <div>
                <h1 className="text-2xl font-bold text-zinc-100">Settings</h1>
                <p className="text-sm text-zinc-400 mt-1">System information and user management</p>
            </div>

            {/* User Info */}
            <Card className="bg-zinc-800/50 border-zinc-700/50">
                <CardHeader>
                    <CardTitle className="text-zinc-200 text-lg">Profile</CardTitle>
                    <CardDescription className="text-zinc-400">Your account information</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-zinc-400">Username</span>
                        <span className="text-sm font-medium text-zinc-200">{user?.username}</span>
                    </div>
                    <Separator className="bg-zinc-700/50" />
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-zinc-400">Role</span>
                        <Badge variant={user?.role === 'admin' ? 'default' : 'secondary'} className="capitalize">
                            {user?.role}
                        </Badge>
                    </div>
                    <Separator className="bg-zinc-700/50" />
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-zinc-400">Organization</span>
                        <span className="text-sm font-medium text-zinc-200">{user?.org_id}</span>
                    </div>
                </CardContent>
            </Card>

            {/* System Health */}
            <Card className="bg-zinc-800/50 border-zinc-700/50">
                <CardHeader>
                    <CardTitle className="text-zinc-200 text-lg">System Status</CardTitle>
                    <CardDescription className="text-zinc-400">Backend health and version</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-zinc-400">API Status</span>
                        <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-emerald-400' : 'bg-red-400'}`} />
                            <span className="text-sm font-medium text-zinc-200">{health?.status || 'Checking...'}</span>
                        </div>
                    </div>
                    <Separator className="bg-zinc-700/50" />
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-zinc-400">Version</span>
                        <span className="text-sm font-medium text-zinc-200">{health?.version || '—'}</span>
                    </div>
                    <Separator className="bg-zinc-700/50" />
                    <div className="flex items-center justify-between">
                        <span className="text-sm text-zinc-400">Milvus Connected</span>
                        <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${health?.milvus_connected ? 'bg-emerald-400' : 'bg-red-400'}`} />
                            <span className="text-sm font-medium text-zinc-200">{health?.milvus_connected ? 'Yes' : 'No'}</span>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Admin: User Management */}
            {isAdmin && (
                <Card className="bg-zinc-800/50 border-zinc-700/50">
                    <CardHeader>
                        <CardTitle className="text-zinc-200 text-lg">Users</CardTitle>
                        <CardDescription className="text-zinc-400">All registered users (admin only)</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {usersError ? (
                            <p className="text-sm text-red-400">{usersError}</p>
                        ) : users.length === 0 ? (
                            <p className="text-sm text-zinc-500">Loading users...</p>
                        ) : (
                            <div className="space-y-2">
                                {users.map(u => (
                                    <div key={u.username} className="flex items-center justify-between py-2">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500/30 to-violet-500/30 flex items-center justify-center">
                                                <span className="text-xs font-bold text-cyan-300 uppercase">
                                                    {u.username.charAt(0)}
                                                </span>
                                            </div>
                                            <div>
                                                <p className="text-sm font-medium text-zinc-200">{u.username}</p>
                                                <p className="text-xs text-zinc-500">org: {u.org_id}</p>
                                            </div>
                                        </div>
                                        <Badge variant={u.role === 'admin' ? 'default' : 'secondary'} className="capitalize">
                                            {u.role}
                                        </Badge>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
