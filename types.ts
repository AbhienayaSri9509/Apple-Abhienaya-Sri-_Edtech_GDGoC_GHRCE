export interface Message {
  role: 'user' | 'model';
  content: string;
  timestamp: Date;
}

export interface AvatarState {
  isThinking: boolean;
  isTalking: boolean;
  mood: 'neutral' | 'happy' | 'thoughtful' | 'excited';
}

export enum ChatStatus {
  IDLE = 'IDLE',
  LOADING = 'LOADING',
  ERROR = 'ERROR'
}