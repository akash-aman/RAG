import { useState, useEffect, useRef, useCallback } from 'react';
import { ingestFile, listDocuments, deleteDocument, deleteAllDocuments, type DocumentInfo } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';

export default function DocumentsPage() {
    const [documents, setDocuments] = useState<DocumentInfo[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isUploading, setIsUploading] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [deleteAllOpen, setDeleteAllOpen] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const fetchDocs = useCallback(async () => {
        try {
            const docs = await listDocuments();
            setDocuments(docs);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load documents');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchDocs();
    }, [fetchDocs]);

    const handleUpload = async (files: FileList | null) => {
        if (!files || files.length === 0) return;
        setIsUploading(true);
        setError('');
        setSuccess('');

        try {
            for (const file of Array.from(files)) {
                await ingestFile(file);
            }
            setSuccess(`Uploaded ${files.length} file${files.length > 1 ? 's' : ''} successfully`);
            await fetchDocs();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setIsUploading(false);
        }
    };

    const handleDelete = async (docId: string) => {
        try {
            await deleteDocument(docId);
            setDocuments(prev => prev.filter(d => d.doc_id !== docId));
            setSuccess('Document deleted');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Delete failed');
        }
    };

    const handleDeleteAll = async () => {
        try {
            await deleteAllDocuments();
            setDocuments([]);
            setSuccess('All documents deleted');
            setDeleteAllOpen(false);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Delete all failed');
        }
    };

    const handleDrag = (e: React.DragEvent, entering: boolean) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(entering);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        handleUpload(e.dataTransfer.files);
    };

    const formatDate = (timestamp: number) => {
        if (!timestamp) return '—';
        return new Date(timestamp * 1000).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });
    };

    return (
        <div className="flex flex-col h-full p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-zinc-100">Documents</h1>
                    <p className="text-sm text-zinc-400 mt-1">
                        Upload and manage your knowledge base documents
                    </p>
                </div>
                {documents.length > 0 && (
                    <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => setDeleteAllOpen(true)}
                        className="cursor-pointer"
                    >
                        Delete All
                    </Button>
                )}
            </div>

            {/* Notifications */}
            {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                    {error}
                </div>
            )}
            {success && (
                <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
                    {success}
                </div>
            )}

            {/* Upload Zone */}
            <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${isDragging
                        ? 'border-cyan-500 bg-cyan-500/10'
                        : 'border-zinc-700 hover:border-zinc-500 bg-zinc-800/30'
                    }`}
                onDragEnter={e => handleDrag(e, true)}
                onDragLeave={e => handleDrag(e, false)}
                onDragOver={e => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.txt,.md"
                    className="hidden"
                    onChange={e => handleUpload(e.target.files)}
                />
                <div className="flex flex-col items-center space-y-3">
                    <div className="w-12 h-12 rounded-xl bg-zinc-700/50 flex items-center justify-center">
                        {isUploading ? (
                            <svg className="w-6 h-6 text-cyan-400 animate-spin" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                        ) : (
                            <svg className="w-6 h-6 text-zinc-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                            </svg>
                        )}
                    </div>
                    <div>
                        <p className="text-sm font-medium text-zinc-300">
                            {isUploading ? 'Uploading...' : 'Drop files here or click to upload'}
                        </p>
                        <p className="text-xs text-zinc-500 mt-1">Supports PDF, TXT, MD</p>
                    </div>
                </div>
            </div>

            {/* Document List */}
            <ScrollArea className="flex-1">
                {isLoading ? (
                    <div className="flex items-center justify-center py-12">
                        <svg className="w-6 h-6 text-zinc-500 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                    </div>
                ) : documents.length === 0 ? (
                    <div className="text-center py-12">
                        <p className="text-zinc-500 text-sm">No documents uploaded yet</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {documents.map(doc => (
                            <Card key={doc.doc_id} className="bg-zinc-800/50 border-zinc-700/50 hover:border-zinc-600/50 transition-colors">
                                <CardHeader className="p-4 pb-2">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1 min-w-0">
                                            <CardTitle className="text-sm font-medium text-zinc-200 truncate">
                                                {doc.source || doc.doc_id}
                                            </CardTitle>
                                            <CardDescription className="text-xs text-zinc-500 mt-1 font-mono">
                                                {doc.doc_id}
                                            </CardDescription>
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => handleDelete(doc.doc_id)}
                                            className="shrink-0 h-8 w-8 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 cursor-pointer"
                                        >
                                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                                            </svg>
                                        </Button>
                                    </div>
                                </CardHeader>
                                <CardContent className="p-4 pt-0">
                                    <div className="flex items-center gap-3 text-xs text-zinc-400">
                                        <Badge variant="secondary" className="bg-zinc-700/50 text-zinc-300">
                                            {doc.chunk_count} chunks
                                        </Badge>
                                        <Separator orientation="vertical" className="h-3" />
                                        <span>{formatDate(doc.created_at)}</span>
                                        {doc.org_id && (
                                            <>
                                                <Separator orientation="vertical" className="h-3" />
                                                <span>org: {doc.org_id}</span>
                                            </>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}
            </ScrollArea>

            {/* Delete All Confirmation Dialog */}
            <Dialog open={deleteAllOpen} onOpenChange={setDeleteAllOpen}>
                <DialogContent className="bg-zinc-900 border-zinc-700">
                    <DialogHeader>
                        <DialogTitle className="text-zinc-100">Delete all documents?</DialogTitle>
                        <DialogDescription className="text-zinc-400">
                            This will permanently delete all {documents.length} document{documents.length !== 1 ? 's' : ''} from the knowledge base. This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setDeleteAllOpen(false)} className="cursor-pointer">
                            Cancel
                        </Button>
                        <Button variant="destructive" onClick={handleDeleteAll} className="cursor-pointer">
                            Delete All
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
