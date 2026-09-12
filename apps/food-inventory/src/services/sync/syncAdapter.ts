export interface SyncAdapter {
  pull(): Promise<void>;
  push(): Promise<void>;
  sync(): Promise<void>;
}

export class PlannedGitHubSyncAdapter implements SyncAdapter {
  private unavailable(): Promise<never> {
    return Promise.reject(new Error("GitHub Sync is planned for a later milestone."));
  }

  pull(): Promise<void> {
    return this.unavailable();
  }

  push(): Promise<void> {
    return this.unavailable();
  }

  sync(): Promise<void> {
    return this.unavailable();
  }
}
