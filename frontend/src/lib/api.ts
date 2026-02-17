const API_BASE = 'http://localhost:8080';

function getToken(): string | null {
    return localStorage.getItem('rag_token');
}

function authHeaders(): HeadersInit {
    const token = getToken();
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    return headers;
}

async function handleResponse<T>(res: Response): Promise<T> {
    if (!res.ok) {
        const body = await res.text();
        let message = `HTTP ${res.status}`;
        try {
            const json = JSON.parse(body);
            message = json.detail || json.message || message;
        } catch { /* ignore */ }
        throw new Error(message);
    }
    return res.json();
}

// ── Auth ────────────────────────────────────────────────────────

export interface TokenResponse {
    access_token: string;
    token_type: string;
}

export interface User {
    username: string;
    role: string;
    org_id: string;
}

export async function login(username: string, password: string): Promise<TokenResponse> {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
    });
    return handleResponse<TokenResponse>(res);
}

export async function register(username: string, password: string, org_id = 'default'): Promise<TokenResponse> {
    const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, org_id }),
    });
    return handleResponse<TokenResponse>(res);
}

export async function getMe(): Promise<User> {
    const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
        headers: authHeaders(),
    });
    return handleResponse<User>(res);
}

export async function getUsers(): Promise<User[]> {
    const res = await fetch(`${API_BASE}/api/v1/auth/users`, {
        headers: authHeaders(),
    });
    return handleResponse<User[]>(res);
}

// ── Ingestion ───────────────────────────────────────────────────

export interface DocumentInfo {
    doc_id: string;
    source: string;
    user_id: string;
    org_id: string;
    chunk_count: number;
    metadata: Record<string, unknown>;
    created_at: number;
}

export interface IngestResponse {
    task_id: string;
    status: string;
    message: string;
}

export async function ingestFile(file: File): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const token = getToken();
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}/api/v1/ingest`, {
        method: 'POST',
        headers,
        body: formData,
    });
    return handleResponse<IngestResponse>(res);
}

export async function listDocuments(): Promise<DocumentInfo[]> {
    const res = await fetch(`${API_BASE}/api/v1/ingest`, {
        headers: authHeaders(),
    });
    return handleResponse<DocumentInfo[]>(res);
}

export async function deleteDocument(docId: string): Promise<{ deleted: number; message: string }> {
    const res = await fetch(`${API_BASE}/api/v1/ingest/${docId}`, {
        method: 'DELETE',
        headers: authHeaders(),
    });
    return handleResponse(res);
}

export async function deleteAllDocuments(): Promise<{ deleted: number; message: string }> {
    const res = await fetch(`${API_BASE}/api/v1/ingest`, {
        method: 'DELETE',
        headers: authHeaders(),
    });
    return handleResponse(res);
}

// ── Query ───────────────────────────────────────────────────────

export interface Source {
    text: string;
    source: string;
    score: number;
    metadata: Record<string, unknown>;
}

export interface QueryResponse {
    answer: string;
    sources: Source[];
    metadata: Record<string, unknown>;
}

export interface QueryOptions {
    filters?: Record<string, unknown>;
    enable_hyde?: boolean;
    enable_reranking?: boolean;
    enable_self_rag?: boolean;
    top_k?: number;
}

export async function query(q: string, opts: QueryOptions = {}): Promise<QueryResponse> {
    const res = await fetch(`${API_BASE}/api/v1/query`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ query: q, ...opts }),
    });
    return handleResponse<QueryResponse>(res);
}

export async function queryStream(
    q: string,
    opts: QueryOptions = {},
    onChunk: (content: string) => void,
    onError?: (error: string) => void,
    onDone?: () => void,
): Promise<void> {
    const res = await fetch(`${API_BASE}/api/v1/query/stream`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ query: q, ...opts }),
    });

    if (!res.ok) {
        const body = await res.text();
        throw new Error(`HTTP ${res.status}: ${body}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        for (const line of chunk.split('\n')) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6);

            if (payload === '[DONE]') {
                onDone?.();
                return;
            }

            try {
                const data = JSON.parse(payload);
                if (data.content) {
                    onChunk(data.content);
                }
                if (data.error) {
                    onError?.(data.error);
                }
            } catch {
                // ignore malformed JSON
            }
        }
    }

    onDone?.();
}

// ── Health ──────────────────────────────────────────────────────

export interface HealthResponse {
    status: string;
    version: string;
    milvus_connected: boolean;
}

export async function healthCheck(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<HealthResponse>(res);
}
