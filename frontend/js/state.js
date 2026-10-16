const HISTORY_LIMIT = 50;

export class EditHistory {
  constructor(document, limit = HISTORY_LIMIT) {
    if (!document || !Array.isArray(document.edits)) {
      throw new TypeError("EditHistory requires a watercolor document.");
    }
    this.document = document;
    this.limit = Math.max(1, Math.floor(limit));
    this.undoStack = [];
    this.redoStack = [];
  }

  addStroke(stroke) {
    this.track(stroke);
    this.document.edits.push(structuredClone(stroke));
    return this.document.edits.at(-1);
  }

  track(stroke) {
    this.undoStack.push(structuredClone(stroke));
    if (this.undoStack.length > this.limit) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  undo() {
    if (this.undoStack.length === 0) return null;
    const stroke = this.undoStack.pop();
    this.redoStack.push(stroke);
    this.document.edits.pop();
    return structuredClone(stroke);
  }

  redo() {
    if (this.redoStack.length === 0) return null;
    const stroke = this.redoStack.pop();
    this.undoStack.push(stroke);
    this.document.edits.push(structuredClone(stroke));
    return structuredClone(stroke);
  }

  clear() {
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.document.edits.length = 0;
  }
}
