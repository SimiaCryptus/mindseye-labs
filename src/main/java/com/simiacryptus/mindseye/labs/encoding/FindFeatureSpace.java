package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;

public abstract class FindFeatureSpace {
  public final int inputBands;
  public final NotebookOutput log;

  public FindFeatureSpace(NotebookOutput log, int inputBands) {
    this.log = log;
    this.inputBands = inputBands;

  }

  @Nonnull
  public abstract FindFeatureSpace invoke();
}
