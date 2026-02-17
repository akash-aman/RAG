import { useState, useRef, useEffect } from 'react';
import { queryStream, type Source } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: Source[];
    isStreaming?: boolean;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = async () => {
        const q = input.trim();
        if (!q || isStreaming) return;

        const userMsg: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: q,
        };

        const assistantId = crypto.randomUUID();
        const assistantMsg: Message = {
            id: assistantId,
            role: 'assistant',
            content: '',
            isStreaming: true,
        };

        setMessages(prev => [...prev, userMsg, assistantMsg]);
        setInput('');
        setIsStreaming(true);

        try {
            await queryStream(
                q,
                {},
                (chunk) => {
                    setMessages(prev =>
                        prev.map(m =>
                            m.id === assistantId
                                ? { ...m, content: m.content + chunk }
                                : m
                        )
                    );
                },
                (error) => {
                    setMessages(prev =>
                        prev.map(m =>
                            m.id === assistantId
                                ? { ...m, content: m.content + `\n\n⚠️ Error: ${error}`, isStreaming: false }
                                : m
                        )
                    );
                },
                () => {
                    setMessages(prev =>
                        prev.map(m =>
                            m.id === assistantId
                                ? { ...m, isStreaming: false }
                                : m
                        )
                    );
                }
            );
        } catch (err) {
            setMessages(prev =>
                prev.map(m =>
                    m.id === assistantId
                        ? { ...m, content: `Error: ${err instanceof Error ? err.message : 'Unknown error'}`, isStreaming: false }
                        : m
                )
            );
        } finally {
            setIsStreaming(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Messages */}
            <ScrollArea className="flex-1 px-4" ref={scrollRef}>
                {messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center space-y-4">
                        <div className="w-16 h-16 rounded-2xl bg-linear-to-br from-cyan-500/20 to-violet-500/20 border border-cyan-500/10 flex items-center justify-center">
                            <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
                            </svg>
                        </div>
                        <div>
                            <h2 className="text-xl font-semibold text-zinc-200 mb-1">Ask your knowledge base</h2>
                            <p className="text-sm text-zinc-500">Upload documents first, then ask questions about them</p>
                        </div>
                    </div>
                ) : (
                    <div className="max-w-3xl mx-auto py-6 space-y-6">
                        {messages.map(msg => (
                            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div
                                    className={`max-w-[85%] rounded-2xl px-4 py-3 ${msg.role === 'user'
                                            ? 'bg-cyan-600/90 text-white'
                                            : 'bg-zinc-800/80 text-zinc-200 border border-zinc-700/50'
                                        }`}
                                >
                                    <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                                    {msg.isStreaming && (
                                        <span className="inline-block w-2 h-4 bg-cyan-400 animate-pulse ml-1 rounded-sm" />
                                    )}
                                    {msg.sources && msg.sources.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-zinc-600/50 space-y-1">
                                            <p className="text-xs font-medium text-zinc-400 mb-1">Sources</p>
                                            {msg.sources.map((s, i) => (
                                                <Badge key={i} variant="secondary" className="text-xs mr-1 bg-zinc-700">
                                                    {s.source || 'Unknown'}
                                                </Badge>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </ScrollArea>

            {/* Input */}
            <div className="border-t border-zinc-800 bg-zinc-900/50 p-4">
                <div className="max-w-3xl mx-auto">
                    <Card className="flex items-end gap-2 p-2 bg-zinc-800/50 border-zinc-700/50">
                        <Textarea
                            ref={textareaRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask a question about your documents..."
                            className="min-h-[44px] max-h-[200px] resize-none bg-transparent border-0 focus-visible:ring-0 text-zinc-100 placeholder:text-zinc-500 text-sm"
                            rows={1}
                            disabled={isStreaming}
                        />
                        <Button
                            onClick={handleSend}
                            disabled={!input.trim() || isStreaming}
                            size="icon"
                            className="shrink-0 bg-cyan-600 hover:bg-cyan-500 text-white h-9 w-9 rounded-lg cursor-pointer"
                        >
                            {isStreaming ? (
                                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                            ) : (
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                                </svg>
                            )}
                        </Button>
                    </Card>
                    <p className="text-xs text-zinc-600 text-center mt-2">
                        Press Enter to send, Shift+Enter for new line
                    </p>
                </div>
            </div>
        </div>
    );
}
