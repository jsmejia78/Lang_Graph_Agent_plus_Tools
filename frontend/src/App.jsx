import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import 'katex/dist/katex.min.css';

// Function to transform bracket-wrapped LaTeX expressions
const transformBracketMath = (text) => {
  if (!text) return text;
  
  // Transform [ ... ] to $ ... $ with precise spacing
  return text.replace(/\[\s*([^\[\]]*?)\s*\]/g, (match, captured) => {
    // Remove all leading and trailing spaces from the captured content
    const cleanContent = captured.trim();
    return `$${cleanContent}$`;
  });
};

// Simple error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: "1rem", 
          background: "#fef2f2", 
          color: "#dc2626", 
          borderRadius: "8px",
          margin: "1rem"
        }}>
          <h3>Something went wrong</h3>
          <p>Please refresh the page and try again.</p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              padding: "0.5rem 1rem",
              background: "#dc2626",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Refresh Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default function App() {
  // State for form fields
  const [SystemMessage, setSystemMessage] = useState("");
  const [userMessage, setUserMessage] = useState("");
  const [model, setModel] = useState("gpt-4.1-mini");
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  // Chat conversation history
  const [conversation, setConversation] = useState([]);
  
  // Health check state
  const [health, setHealth] = useState(null);
  
  // Data File RAG state
  const [dataFiles, setDataFiles] = useState([]);
  const [dataFileStatus, setDataFileStatus] = useState(null);
  const [isRagMode, setIsRagMode] = useState(false);
  const [uploadingDataFile, setUploadingDataFile] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0, fileName: '' });
  const [showFileInfoPopup, setShowFileInfoPopup] = useState(false);
  const [showFileHistoryPopup, setShowFileHistoryPopup] = useState(false);
  const [showSettingsPopup, setShowSettingsPopup] = useState(false);
  const [fileHistory, setFileHistory] = useState([]);
  
  // Ref for auto-scrolling to latest message
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (chatEndRef.current && chatContainerRef.current) {
      // Use a more reliable scroll method
      const scrollToBottom = () => {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      };
      
      // Use requestAnimationFrame to ensure DOM is updated
      requestAnimationFrame(scrollToBottom);
    }
  }, [conversation]);

  // Force scroll to bottom when loading state changes
  useEffect(() => {
    if (chatEndRef.current && chatContainerRef.current) {
      const scrollToBottom = () => {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      };
      requestAnimationFrame(scrollToBottom);
    }
  }, [loading]);

  // Health check on mount
  React.useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => setHealth(data.status === "ok" ? "🟢" : "🔴"))
      .catch(() => setHealth("🔴"));
  }, []);

  // Check Data File status on mount
  React.useEffect(() => {
    fetch("/api/data-file-indexing-status")
      .then((res) => res.json())
      .then((data) => {
        console.log('Initial status check:', data); // Debug log
        setDataFileStatus(data);
        // If there's no file history but backend shows indexed, clear it
        if (data.is_indexed && fileHistory.length === 0) {
          // Clear the backend index since user doesn't have a file selected
          fetch("/api/clear-data-file-index", { method: "DELETE" })
            .then(() => {
              setDataFileStatus({ is_indexed: false, chunks_count: 0, file_info: null });
              setIsRagMode(false);
            })
            .catch(() => {
              // If clearing fails, just reset frontend state
              setDataFileStatus({ is_indexed: false, chunks_count: 0, file_info: null });
              setIsRagMode(false);
            });
        } else {
          // Set RAG mode based on actual indexing status
          setIsRagMode(data.is_indexed || false);
        }
      })
      .catch(() => {
        setDataFileStatus({ is_indexed: false, chunks_count: 0, file_info: null });
        setIsRagMode(false);
      });
  }, [dataFiles]);

  // Fetch file history on mount
  React.useEffect(() => {
    fetchFileHistory();
  }, []);

  // Handle form submit
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userMessage.trim()) return;
    
    // Use RAG chat if data file is indexed, otherwise use regular chat
    if (isRagMode && dataFileStatus?.is_indexed) {
      await handleRagChat(e);
    } else {
      await handleRegularChat(e);
    }
  };

  // Handle regular chat
  const handleRegularChat = async (e) => {
    e.preventDefault();
    if (!userMessage.trim()) return;
    
    setLoading(true);
    setError("");
    
    // Add user message to conversation
    const userMsg = { type: "user", content: userMessage, timestamp: new Date() };
    setConversation(prev => [...prev, userMsg]);
    
    const currentUserMessage = userMessage;
    setUserMessage(""); // Clear input immediately
    
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          system_message: SystemMessage,
          user_message: currentUserMessage,
          model,
          api_key: apiKey,
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      let result = "";
      
      // Add assistant message placeholder
      const assistantMsg = { type: "assistant", content: "", timestamp: new Date() };
      setConversation(prev => [...prev, assistantMsg]);
      
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        result += chunk;
        
        // Update only the last message content more efficiently
        setConversation(prev => {
          const newConversation = [...prev];
          const lastIndex = newConversation.length - 1;
          if (lastIndex >= 0 && newConversation[lastIndex].type === "assistant") {
            newConversation[lastIndex] = {
              ...newConversation[lastIndex],
              content: result
            };
          }
          return newConversation;
        });
      }
    } catch (err) {
      console.error('Chat error:', err);
      setError(err.message || "Unknown error occurred");
      // Remove the assistant message if there was an error
      setConversation(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // Clear conversation
  const clearConversation = () => {
    setConversation([]);
    setError("");
  };

  // Handle Data File selection
  const handleFileChange = (e) => {
    if (!apiKey.trim()) {
      setError("Please enter your API key first before selecting files");
      return;
    }
    
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      // Files were actually selected
      const validFiles = files.filter(file => 
        file.type === "application/pdf" || file.type === "text/plain"
      );
      
      if (validFiles.length === files.length) {
        // All files are valid
        setDataFiles(validFiles);
        setError("");
      } else if (validFiles.length > 0) {
        // Some files are valid, some aren't
        setDataFiles(validFiles);
        setError(`${files.length - validFiles.length} invalid file(s) skipped. Only PDF and TXT files are supported.`);
      } else {
        // No valid files
        setError("Please select valid PDF or TXT files");
        setDataFiles([]);
      }
    } else {
      // User canceled file selection - clear any previously selected files
      setDataFiles([]);
      setError("");
    }
  };

  // Upload and index Data Files (one by one)
  const uploadDataFiles = async () => {
    if (dataFiles.length === 0 || !apiKey) {
      setError("Please select PDF or TXT files and enter your API key");
      return;
    }

    setUploadingDataFile(true);
    setError("");
    setUploadProgress({ current: 0, total: dataFiles.length, fileName: '' });

    try {
      for (let i = 0; i < dataFiles.length; i++) {
        const file = dataFiles[i];
        setUploadProgress({ current: i + 1, total: dataFiles.length, fileName: file.name });

        const formData = new FormData();
        formData.append("file", file);
        formData.append("api_key", apiKey);

        const response = await fetch("/api/upload-data-file", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(`Failed to upload ${file.name}: ${errorData.detail || "Unknown error"}`);
        }

        const result = await response.json();
        console.log('Upload result:', result); // Debug log
        
        // Update status with the latest uploaded file
        setDataFileStatus({
          is_indexed: true,
          document_id: result.document_id,
          chunks_count: result.chunks_count,
          file_info: result.file_info
        });
        setIsRagMode(true);
        
        // Immediately update file history for each uploaded file
        const newHistoryEntry = {
          document_id: result.document_id,
          filename: result.file_info.filename,
          file_type: result.file_info.file_type,
          file_size: result.file_info.file_size,
          upload_timestamp: result.file_info.upload_timestamp,
          chunks_count: result.chunks_count,
          is_current: i === dataFiles.length - 1 // Only the last file is current
        };
        
        // Mark all previous files as not current and add new file
        setFileHistory(prev => {
          const updated = prev.map(file => ({ ...file, is_current: false }));
          updated.push(newHistoryEntry);
          return updated;
        });
      }

      setError("");
      setDataFiles([]); // Clear selected files after successful upload
      
      // Clear the file input field
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
    } catch (err) {
      console.error("Data file upload error:", err);
      setError(err.message || "Failed to upload files");
    } finally {
      setUploadingDataFile(false);
      setUploadProgress({ current: 0, total: 0, fileName: '' });
    }
  };

  // Clear indexed data file
  const clearDataFileIndex = async () => {
    try {
      await fetch("/api/clear-data-file-index", { method: "DELETE" });
      setDataFileStatus({ is_indexed: false, chunks_count: 0, file_info: null });
      setDataFiles([]);
      setIsRagMode(false);
      setError("");
      
      // Clear the file input field
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      // Immediately clear file history to avoid button lag
      setFileHistory([]);
    } catch (err) {
      console.error("Error clearing data file index:", err);
      setError("Failed to clear data file index");
    }
  };

  // Function to format file size
  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  // Function to format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown';
    return new Date(timestamp).toLocaleString();
  };

  // Function to fetch file history
  const fetchFileHistory = async () => {
    try {
      const response = await fetch("/api/file-history");
      const data = await response.json();
      if (data.success) {
        setFileHistory(data.file_history || []);
      } else {
        setFileHistory([]);
      }
    } catch (err) {
      console.error("Error fetching file history:", err);
      setFileHistory([]);
    }
  };

  // Function to show file history popup
  const showFileHistoryModal = () => {
    // No need to fetch again since we maintain local state
    setShowFileHistoryPopup(true);
  };

  // Get current file info for display (from dataFileStatus or latest from history)
  const getCurrentFileInfo = () => {
    if (dataFileStatus?.file_info) {
      return {
        filename: dataFileStatus.file_info.filename,
        file_type: dataFileStatus.file_info.file_type,
        file_size: dataFileStatus.file_info.file_size,
        upload_timestamp: dataFileStatus.file_info.upload_timestamp,
        document_id: dataFileStatus.document_id,
        chunks_count: dataFileStatus.chunks_count
      };
    }
    
    // Fallback to latest file from history
    const currentFile = fileHistory.find(f => f.is_current);
    if (currentFile) {
      return {
        filename: currentFile.filename,
        file_type: currentFile.file_type,
        file_size: currentFile.file_size,
        upload_timestamp: currentFile.upload_timestamp,
        document_id: currentFile.document_id,
        chunks_count: currentFile.chunks_count
      };
    }
    
    return null;
  };

  // Handle RAG chat
  const handleRagChat = async (e) => {
    e.preventDefault();
    if (!userMessage.trim() || !dataFileStatus?.is_indexed) return;
    
    setLoading(true);
    setError("");
    
    // Add user message to conversation
    const userMsg = { type: "user", content: userMessage, timestamp: new Date() };
    setConversation(prev => [...prev, userMsg]);
    
    const currentUserMessage = userMessage;
    setUserMessage(""); // Clear input immediately
    
    try {
      const res = await fetch("/api/rag-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: currentUserMessage,
          system_message: SystemMessage,
          api_key: apiKey,
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      let result = "";
      
      // Add assistant message placeholder
      const assistantMsg = { type: "assistant", content: "", timestamp: new Date() };
      setConversation(prev => [...prev, assistantMsg]);
      
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        result += chunk;
        
        // Update only the last message content more efficiently
        setConversation(prev => {
          const newConversation = [...prev];
          const lastIndex = newConversation.length - 1;
          if (lastIndex >= 0 && newConversation[lastIndex].type === "assistant") {
            newConversation[lastIndex] = {
              ...newConversation[lastIndex],
              content: result
            };
          }
          return newConversation;
        });
      }
    } catch (err) {
      console.error('RAG chat error:', err);
      setError(err.message || "Unknown error occurred");
      // Remove the assistant message if there was an error
      setConversation(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // Custom styles for markdown components with tighter spacing
  const markdownComponents = {
    // Code blocks
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      return !inline ? (
        <pre style={{
          background: '#1f2937',
          color: '#f9fafb',
          padding: '0.75rem',
          borderRadius: '6px',
          overflow: 'auto',
          margin: '0.25rem 0',
          fontSize: '0.875rem',
          lineHeight: '1.4'
        }}>
          <code className={className} {...props}>
            {children}
          </code>
        </pre>
      ) : (
        <code style={{
          background: '#f3f4f6',
          color: '#dc2626',
          padding: '0.125rem 0.25rem',
          borderRadius: '4px',
          fontSize: '0.875em',
          fontFamily: 'monospace'
        }} {...props}>
          {children}
        </code>
      );
    },
    // Headers with tighter spacing
    h1: ({ children }) => <h1 style={{ fontSize: '1.25rem', fontWeight: '600', margin: '0.5rem 0 0.25rem 0', color: '#111827' }}>{children}</h1>,
    h2: ({ children }) => <h2 style={{ fontSize: '1.125rem', fontWeight: '600', margin: '0.375rem 0 0.25rem 0', color: '#111827' }}>{children}</h2>,
    h3: ({ children }) => <h3 style={{ fontSize: '1rem', fontWeight: '600', margin: '0.25rem 0 0.125rem 0', color: '#111827' }}>{children}</h3>,
    // Lists with tighter spacing
    ul: ({ children }) => <ul style={{ margin: '0.25rem 0', paddingLeft: '1.25rem' }}>{children}</ul>,
    ol: ({ children }) => <ol style={{ margin: '0.25rem 0', paddingLeft: '1.25rem' }}>{children}</ol>,
    li: ({ children }) => <li style={{ margin: '0.125rem 0', display: 'list-item', whiteSpace: 'normal' }}>{children}</li>,
    // Links
    a: ({ href, children }) => (
      <a 
        href={href} 
        target="_blank" 
        rel="noopener noreferrer"
        style={{ 
          color: '#3b82f6', 
          textDecoration: 'underline',
          wordBreak: 'break-word'
        }}
      >
        {children}
      </a>
    ),
    // Blockquotes with tighter spacing
    blockquote: ({ children }) => (
      <blockquote style={{
        borderLeft: '3px solid #3b82f6',
        margin: '0.25rem 0',
        padding: '0.25rem 0.75rem',
        background: '#f8fafc',
        borderRadius: '0 4px 4px 0',
        fontStyle: 'italic'
      }}>
        {children}
      </blockquote>
    ),
    // Tables with tighter spacing
    table: ({ children }) => (
      <div style={{ overflow: 'auto', margin: '0.25rem 0' }}>
        <table style={{
          borderCollapse: 'collapse',
          width: '100%',
          fontSize: '0.875rem'
        }}>
          {children}
        </table>
      </div>
    ),
    th: ({ children }) => (
      <th style={{
        border: '1px solid #d1d5db',
        padding: '0.375rem',
        background: '#f9fafb',
        fontWeight: '600',
        textAlign: 'left'
      }}>
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td style={{
        border: '1px solid #d1d5db',
        padding: '0.375rem'
      }}>
        {children}
      </td>
    ),
    // Paragraphs with tighter spacing
    p: ({ children }) => <p style={{ margin: '0.25rem 0', lineHeight: '1.4' }}>{children}</p>,
    // Strong text
    strong: ({ children }) => <strong style={{ fontWeight: '600' }}>{children}</strong>,
    // Emphasis
    em: ({ children }) => <em style={{ fontStyle: 'italic' }}>{children}</em>,
  };

  return (
    <ErrorBoundary>
      {/* Settings Popup Modal */}
      {showSettingsPopup && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            maxWidth: '500px',
            width: '90%',
            maxHeight: '80vh',
            overflowY: 'auto',
            boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
            fontFamily: 'sans-serif'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.5rem'
            }}>
              <h2 style={{
                margin: 0,
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#374151'
              }}>
                ⚙️ Settings
              </h2>
              <button
                onClick={() => setShowSettingsPopup(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  color: '#6b7280',
                  padding: '0.25rem'
                }}
              >
                ×
              </button>
            </div>
            
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem'
            }}>
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  color: '#374151',
                  marginBottom: '0.5rem'
                }}>
                  OpenAI API Key *
                </label>
                <input
                  type="password"
                  placeholder="Enter your OpenAI API Key"
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                    background: '#fff',
                    boxSizing: 'border-box'
                  }}
                />
              </div>
              
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  color: '#374151',
                  marginBottom: '0.5rem'
                }}>
                  System Message (Optional)
                </label>
                <textarea
                  placeholder="Enter system instructions for the AI..."
                  value={SystemMessage}
                  onChange={e => setSystemMessage(e.target.value)}
                  rows={3}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                    background: '#fff',
                    boxSizing: 'border-box',
                    resize: 'vertical'
                  }}
                />
              </div>
              
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  color: '#374151',
                  marginBottom: '0.5rem'
                }}>
                  Model (Optional)
                </label>
                <input
                  type="text"
                  placeholder="gpt-4.1-mini"
                  value={model}
                  onChange={e => setModel(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '0.875rem',
                    background: '#fff',
                    boxSizing: 'border-box'
                  }}
                />
              </div>
              
              <div style={{
                padding: '1rem',
                backgroundColor: '#f0f9ff',
                borderRadius: '8px',
                border: '1px solid #bae6fd'
              }}>
                <div style={{
                  fontSize: '0.875rem',
                  color: '#0369a1',
                  fontWeight: '500'
                }}>
                  💡 Tips
                </div>
                <ul style={{
                  fontSize: '0.75rem',
                  color: '#0284c7',
                  marginTop: '0.5rem',
                  paddingLeft: '1rem',
                  lineHeight: '1.4'
                }}>
                  <li>API Key is required for all functionality</li>
                  <li>System Message helps guide AI behavior</li>
                  <li>Model defaults to gpt-4.1-mini if not specified</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* File Info Popup Modal */}
      {showFileInfoPopup && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            maxWidth: '500px',
            width: '90%',
            maxHeight: '80vh',
            overflowY: 'auto',
            boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
            fontFamily: 'sans-serif'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.5rem'
            }}>
              <h2 style={{
                margin: 0,
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#374151'
              }}>
                📄 Uploaded File Information
              </h2>
              <button
                onClick={() => setShowFileInfoPopup(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  color: '#6b7280',
                  padding: '0.25rem'
                }}
              >
                ×
              </button>
            </div>
            
            {(() => {
              const currentFileInfo = getCurrentFileInfo();
              return currentFileInfo ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1rem'
                }}>
                  <div style={{
                    padding: '1rem',
                    backgroundColor: '#f8fafc',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0'
                  }}>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 2fr',
                      gap: '0.75rem',
                      fontSize: '0.875rem'
                    }}>
                      <div style={{ fontWeight: '600', color: '#374151' }}>Filename:</div>
                      <div style={{ color: '#6b7280', wordBreak: 'break-all' }}>
                        {currentFileInfo.filename}
                      </div>
                      
                      <div style={{ fontWeight: '600', color: '#374151' }}>File Type:</div>
                      <div style={{ color: '#6b7280' }}>
                        {currentFileInfo.file_type}
                      </div>
                      
                      <div style={{ fontWeight: '600', color: '#374151' }}>File Size:</div>
                      <div style={{ color: '#6b7280' }}>
                        {formatFileSize(currentFileInfo.file_size)}
                      </div>
                      
                      <div style={{ fontWeight: '600', color: '#374151' }}>Upload Time:</div>
                      <div style={{ color: '#6b7280' }}>
                        {formatTimestamp(currentFileInfo.upload_timestamp)}
                      </div>
                      
                      <div style={{ fontWeight: '600', color: '#374151' }}>Document ID:</div>
                      <div style={{ color: '#6b7280', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                        {currentFileInfo.document_id}
                      </div>
                      
                      <div style={{ fontWeight: '600', color: '#374151' }}>Segments Count:</div>
                      <div style={{ color: '#6b7280' }}>
                        {currentFileInfo.chunks_count} segments
                      </div>
                    </div>
                  </div>
                  
                  <div style={{
                    padding: '1rem',
                    backgroundColor: '#f0f9ff',
                    borderRadius: '8px',
                    border: '1px solid #bae6fd'
                  }}>
                    <div style={{
                      fontSize: '0.875rem',
                      color: '#0369a1',
                      fontWeight: '500'
                    }}>
                      💡 RAG Information
                    </div>
                    <div style={{
                      fontSize: '0.75rem',
                      color: '#0284c7',
                      marginTop: '0.5rem',
                      lineHeight: '1.4'
                    }}>
                      Your document has been processed and split into {currentFileInfo.chunks_count} text segments. 
                      Each segment contains approximately 1000 characters with 200 characters overlap for context preservation.
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{
                  textAlign: 'center',
                  padding: '2rem',
                  color: '#6b7280'
                }}>
                  No file information available
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {/* File History Popup Modal */}
      {showFileHistoryPopup && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
                     <div style={{
             backgroundColor: 'white',
             padding: '2rem',
             borderRadius: '12px',
             maxWidth: '600px',
             width: '90%',
             maxHeight: '80vh',
             overflowY: 'auto',
             boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
             fontFamily: 'sans-serif'
           }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.5rem'
            }}>
              <h2 style={{
                margin: 0,
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#374151'
              }}>
                📁 Files History
              </h2>
              <button
                onClick={() => setShowFileHistoryPopup(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  color: '#6b7280',
                  padding: '0.25rem'
                }}
              >
                ×
              </button>
            </div>
            
            {fileHistory.length > 0 ? (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '1rem'
              }}>
                {fileHistory.map((file, index) => (
                  <div key={file.document_id} style={{
                    padding: '1rem',
                    backgroundColor: file.is_current ? '#f0f9ff' : '#f8fafc',
                    borderRadius: '8px',
                    border: file.is_current ? '2px solid #3b82f6' : '1px solid #e2e8f0',
                    position: 'relative'
                  }}>
                                         {file.is_current && (
                       <div style={{
                         position: 'absolute',
                         top: '0.5rem',
                         right: '0.5rem',
                         background: '#3b82f6',
                         color: 'white',
                         padding: '0.25rem 0.5rem',
                         borderRadius: '12px',
                         fontSize: '0.75rem',
                         fontWeight: '500'
                       }}>
                         Latest
                       </div>
                     )}
                    
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr auto auto',
                      gap: '1rem',
                      alignItems: 'center'
                    }}>
                      <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.25rem'
                      }}>
                        <div style={{
                          fontSize: '0.875rem',
                          fontWeight: '600',
                          color: '#374151',
                          wordBreak: 'break-all'
                        }}>
                          {file.filename}
                        </div>
                        <div style={{
                          fontSize: '0.75rem',
                          color: '#6b7280'
                        }}>
                          Uploaded: {formatTimestamp(file.upload_timestamp)}
                        </div>
                      </div>
                      
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.25rem 0.5rem',
                        backgroundColor: file.file_type === 'PDF' ? '#fef3c7' : '#dbeafe',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        fontWeight: '500',
                        color: file.file_type === 'PDF' ? '#92400e' : '#1e40af'
                      }}>
                        <span>{file.file_type === 'PDF' ? '📄' : '📝'}</span>
                        {file.file_type}
                      </div>
                      
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.25rem 0.5rem',
                        backgroundColor: '#f3f4f6',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        fontWeight: '500',
                        color: '#374151'
                      }}>
                        <span>🔢</span>
                        {file.chunks_count} segments
                      </div>
                    </div>
                  </div>
                ))}
                
                <div style={{
                  padding: '1rem',
                  backgroundColor: '#f9fafb',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb',
                  textAlign: 'center',
                  fontSize: '0.875rem',
                  color: '#6b7280'
                }}>
                  Total Files: {fileHistory.length}
                </div>
              </div>
            ) : (
              <div style={{
                textAlign: 'center',
                padding: '3rem 2rem',
                color: '#6b7280'
              }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📁</div>
                <div style={{ fontSize: '1rem', fontWeight: '500' }}>No files uploaded yet</div>
                <div style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                  Upload your first PDF or TXT file to get started
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      <div className="app-container" style={{ 
        fontFamily: "sans-serif", 
        maxWidth: "100%", 
        margin: "0", 
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#fff",
        borderRadius: 0,
        boxShadow: "none",
        overflow: "hidden"
      }}>
        {/* Desktop Container - only applies on larger screens */}
        <div style={{
          maxWidth: "800px",
          margin: "0 auto",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "#fff",
          borderRadius: "16px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
          overflow: "hidden",
          width: "100%"
        }}>
          {/* Header */}
          <div className="header" style={{ 
            padding: "1rem 1.5rem", 
            borderBottom: "1px solid #e1e8ed",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "0.5rem"
          }}>
            <h1 style={{ 
              margin: 0, 
              fontSize: "1.5rem", 
              color: "#fff", 
              fontWeight: "600",
              flex: 1,
              minWidth: "200px"
            }}>
              AI Chat {health && <span title="API Health">{health}</span>}
              <div style={{ 
                fontSize: "0.875rem", 
                marginTop: "0.25rem", 
                opacity: 0.9,
                fontWeight: "500"
              }}>
                {isRagMode && dataFileStatus?.is_indexed ? "RAG Chat Mode" : "Standard Chat Mode"}
              </div>
            </h1>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button 
                onClick={() => setShowSettingsPopup(true)}
                style={{
                  padding: "0.5rem 1rem",
                  background: "rgba(255,255,255,0.2)",
                  color: "white",
                  border: "1px solid rgba(255,255,255,0.3)",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "0.875rem",
                  fontWeight: "500",
                  transition: "all 0.2s ease",
                  whiteSpace: "nowrap"
                }}
                onMouseOver={(e) => e.target.style.background = "rgba(255,255,255,0.3)"}
                onMouseOut={(e) => e.target.style.background = "rgba(255,255,255,0.2)"}
              >
                ⚙️ Settings
              </button>
              <button 
                onClick={clearConversation}
                style={{
                  padding: "0.5rem 1rem",
                  background: "rgba(255,255,255,0.2)",
                  color: "white",
                  border: "1px solid rgba(255,255,255,0.3)",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "0.875rem",
                  fontWeight: "500",
                  transition: "all 0.2s ease",
                  whiteSpace: "nowrap"
                }}
                onMouseOver={(e) => e.target.style.background = "rgba(255,255,255,0.3)"}
                onMouseOut={(e) => e.target.style.background = "rgba(255,255,255,0.2)"}
              >
                Clear Chat
              </button>
            </div>
          </div>

          {/* Data File Upload Section */}
          <div style={{ 
            background: "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)", 
            padding: "1rem 1.5rem",
            borderBottom: "1px solid #e1e8ed"
          }}>
            <div style={{ fontSize: "0.9rem", fontWeight: "600", color: "#4338ca", marginBottom: "0.75rem" }}>
              Data File Upload & RAG
            </div>
            <div style={{ 
              display: "flex", 
              alignItems: "center",
              gap: "0.5rem",
              flexWrap: "wrap"
            }}>
              <input
                type="file"
                accept=".pdf,.txt"
                multiple
                onChange={handleFileChange}
                ref={fileInputRef}
                disabled={!apiKey.trim()}
                style={{
                  fontSize: "0.75rem",
                  minWidth: "400px",
                  maxWidth: "500px",
                  opacity: !apiKey.trim() ? 0.5 : 1,
                  cursor: !apiKey.trim() ? "not-allowed" : "pointer"
                }}
              />
              <button
                onClick={uploadDataFiles}
                disabled={dataFiles.length === 0 || !apiKey.trim() || uploadingDataFile}
                style={{
                  padding: "0.5rem 0.75rem",
                  background: (dataFiles.length === 0 || !apiKey.trim() || uploadingDataFile) ? "#94a3b8" : "#3b82f6",
                  color: "white",
                  border: "none",
                  borderRadius: "6px",
                  cursor: (dataFiles.length === 0 || !apiKey.trim() || uploadingDataFile) ? "not-allowed" : "pointer",
                  fontSize: "0.75rem",
                  fontWeight: "500"
                }}
              >
                {uploadingDataFile ? (
                  uploadProgress.total > 1 ? 
                    `Uploading ${uploadProgress.current}/${uploadProgress.total}` : 
                    "Uploading..."
                ) : (
                  dataFiles.length > 1 ? `Upload ${dataFiles.length} Files` : "Upload"
                )}
              </button>
              {fileHistory.length > 0 && (
                <>
                  <button
                    onClick={() => setShowFileInfoPopup(true)}
                    style={{
                      padding: "0.5rem 0.75rem",
                      background: "#059669",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "0.75rem",
                      fontWeight: "500"
                    }}
                  >
                    View Info
                  </button>
                  <button
                    onClick={showFileHistoryModal}
                    style={{
                      padding: "0.5rem 0.75rem",
                      background: "#6366f1",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "0.75rem",
                      fontWeight: "500"
                    }}
                  >
                    Files List
                  </button>
                  <button
                    onClick={clearDataFileIndex}
                    style={{
                      padding: "0.5rem 0.75rem",
                      background: "#ef4444",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "0.75rem",
                      fontWeight: "500"
                    }}
                  >
                    Clear
                  </button>
                </>
              )}
            </div>
            {!apiKey.trim() && (
              <div style={{ 
                fontSize: "0.7rem", 
                color: "#6b7280",
                marginTop: "0.5rem"
              }}>
                💡 Configure your API key in Settings to enable file upload
              </div>
            )}
            {dataFiles.length > 0 && !uploadingDataFile && (
              <div style={{ 
                fontSize: "0.7rem", 
                color: "#374151",
                marginTop: "0.5rem"
              }}>
                📁 {dataFiles.length} file{dataFiles.length > 1 ? 's' : ''} selected: {dataFiles.map(f => f.name).join(', ')}
              </div>
            )}
            {uploadingDataFile && uploadProgress.total > 0 && (
              <div style={{ 
                fontSize: "0.7rem", 
                color: "#3b82f6",
                marginTop: "0.5rem",
                fontWeight: "500"
              }}>
                ⏳ Uploading {uploadProgress.current}/{uploadProgress.total}: {uploadProgress.fileName}
              </div>
            )}
          </div>

          {/* Chat Messages */}
          <div 
            ref={chatContainerRef}
            className="chat-messages"
            style={{ 
              flex: 1, 
              overflowY: "auto", 
              padding: "1rem 1.5rem",
              background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
              height: "calc(90vh - 280px)",
              minHeight: "300px",
              display: "flex",
              flexDirection: "column"
            }}
          >
            <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
              {conversation.length === 0 ? (
                <div style={{ 
                  textAlign: "center", 
                  color: "#64748b", 
                  marginTop: "2rem",
                  fontSize: "1.1rem",
                  fontWeight: "500"
                }}>
                  Start a conversation by typing a message below!
                </div>
              ) : (
                conversation.map((message, index) => (
                  <div
                    key={index}
                    style={{
                      marginBottom: "1rem",
                      display: "flex",
                      justifyContent: message.type === "user" ? "flex-end" : "flex-start"
                    }}
                  >
                    <div
                      style={{
                        maxWidth: "85%",
                        padding: "0.875rem 1.125rem",
                        borderRadius: "20px",
                        background: message.type === "user" 
                          ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" 
                          : "#fff",
                        color: message.type === "user" ? "#fff" : "#374151",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                        wordWrap: "break-word",
                        border: message.type === "assistant" ? "1px solid #e5e7eb" : "none"
                      }}
                    >
                      <div style={{ 
                        marginBottom: "0.25rem", 
                        fontSize: "0.75rem", 
                        opacity: 0.8,
                        fontWeight: "500"
                      }}>
                        {message.type === "user" ? "You" : "AI Assistant"}
                      </div>
                      {message.type === "user" ? (
                        <div style={{ whiteSpace: "pre-wrap" }}>{message.content}</div>
                      ) : (
                        <div style={{ 
                          fontSize: "0.875rem",
                          lineHeight: "1.4",
                          overflow: "auto",
                          maxWidth: "100%"
                        }}>
                          <style>{`
                            .katex-display {
                              margin: 0.75rem 0 !important;
                              overflow-x: auto !important;
                              max-width: 100% !important;
                              text-align: center !important;
                            }
                            .katex {
                              font-size: 1.1em !important;
                              line-height: 1.2 !important;
                            }
                            .katex .mfrac {
                              margin: 0 0.2em !important;
                            }
                          `}</style>
                          {message.content ? (
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm, remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                              components={markdownComponents}
                            >
                              {transformBracketMath(message.content)}
                            </ReactMarkdown>
                          ) : (
                            <div>Loading...</div>
                          )}
                          {loading && index === conversation.length - 1 && (
                            <span style={{ opacity: 0.7 }}>...</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
            <div ref={chatEndRef} style={{ height: "1px" }} />
          </div>

          {/* Error Display */}
          {error && (
            <div style={{ 
              padding: "0.75rem 1.5rem", 
              background: "linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)", 
              color: "#dc2626", 
              borderTop: "1px solid #fca5a5",
              fontSize: "0.875rem",
              fontWeight: "500"
            }}>
              {error}
            </div>
          )}

          {/* Message Input */}
          <div className="message-input" style={{ 
            padding: "1rem 1.5rem", 
            borderTop: "1px solid #e1e8ed",
            background: "#fff"
          }}>
            <form onSubmit={handleSubmit} style={{ display: "flex", gap: "0.75rem" }}>
              <input
                type="text"
                placeholder={isRagMode && dataFileStatus?.is_indexed ? "Ask about your data file..." : "Type your message..."}
                value={userMessage}
                onChange={e => setUserMessage(e.target.value)}
                disabled={loading}
                style={{ 
                  flex: 1, 
                  padding: "0.875rem 1.25rem", 
                  border: "2px solid #e2e8f0", 
                  borderRadius: "25px",
                  fontSize: "1rem",
                  background: "#fff",
                  transition: "border-color 0.2s ease"
                }}
                onFocus={(e) => e.target.style.borderColor = "#667eea"}
                onBlur={(e) => e.target.style.borderColor = "#e2e8f0"}
              />
              <button 
                type="submit" 
                disabled={loading || !userMessage.trim() || !apiKey.trim()} 
                style={{ 
                  padding: "0.875rem 1.5rem", 
                  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", 
                  color: "white", 
                  border: "none", 
                  borderRadius: "25px",
                  cursor: "pointer",
                  fontWeight: "600",
                  fontSize: "1rem",
                  transition: "all 0.2s ease",
                  boxShadow: "0 4px 12px rgba(102, 126, 234, 0.4)",
                  opacity: (loading || !userMessage.trim() || !apiKey.trim()) ? 0.5 : 1
                }}
                onMouseOver={(e) => e.target.style.transform = "translateY(-1px)"}
                onMouseOut={(e) => e.target.style.transform = "translateY(0)"}
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </form>
          </div>

          {/* Footer */}
          <div className="footer" style={{ 
            padding: "0.5rem 1.5rem", 
            fontSize: "0.75rem", 
            color: "#64748b",
            textAlign: "center",
            borderTop: "1px solid #e1e8ed",
            background: "#f8fafc"
          }}>
            Powered by OpenAI & FastAPI | <a href="https://vercel.com/" target="_blank" rel="noopener noreferrer" style={{color: "#667eea", textDecoration: "none"}}>Vercel Ready</a>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
} 