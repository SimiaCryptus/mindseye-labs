/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public abstract class AllTrainingTests extends NotebookReportBase {
  protected final FwdNetworkFactory fwdFactory;
  protected final OptimizationStrategy optimizationStrategy;
  protected final RevNetworkFactory revFactory;
  protected int timeoutMinutes = 10;
  protected int batchSize = 1000;

  public AllTrainingTests(final FwdNetworkFactory fwdFactory, final RevNetworkFactory revFactory,
                          final OptimizationStrategy optimizationStrategy) {
    this.fwdFactory = fwdFactory;
    this.revFactory = revFactory;
    this.optimizationStrategy = optimizationStrategy;
  }

  @Nonnull
  public abstract ImageProblemData getData();

  @Nonnull
  public abstract CharSequence getDatasetName();

  @Nullable
  public static @SuppressWarnings("unused")
  AllTrainingTests[][] addRef(@Nullable AllTrainingTests[][] array) {
    return RefUtil.addRef(array);
  }

  @Test
  @Disabled
  @Tag("Report")
  public void autoencoder_test() {
    @Nonnull NotebookOutput log = getLog();
    log.h1(getDatasetName() + " Denoising Autoencoder");
    intro(log);
    new AutoencodingProblem(fwdFactory, optimizationStrategy, revFactory, getData(), 100, 0.8)
        .setTimeoutMinutes(timeoutMinutes).run(log);
  }

  @Test
  @Tag("Report")
  public void classification_test() {
    @Nonnull NotebookOutput log = getLog();
    log.h1(getDatasetName() + " Denoising Autoencoder");
    intro(log);
    new ClassifyProblem(fwdFactory, optimizationStrategy, getData(), 100).setBatchSize(batchSize)
        .setTimeoutMinutes(timeoutMinutes).run(log);
  }

  @Test
  @Disabled
  @Tag("Report")
  public void encoding_test() {
    @Nonnull NotebookOutput log = getLog();
    log.h1(getDatasetName() + " Image-to-Vector Encoding");
    intro(log);
    new EncodingProblem(revFactory, optimizationStrategy, getData(), 10).setTimeoutMinutes(timeoutMinutes).run(log);
  }

  @Override
  public void printHeader(@Nonnull NotebookOutput log) {
    @Nullable
    CharSequence fwdFactory_javadoc = setReportType(log, fwdFactory.getClass(), "fwd");
    @Nullable
    CharSequence optimizationStrategy_javadoc = setReportType(log, optimizationStrategy.getClass(), "opt");
    @Nullable
    CharSequence revFactory_javadoc = setReportType(log, revFactory.getClass(), "rev");
    super.printHeader(log);
    log.p("_Forward Strategy Javadoc_: " + fwdFactory_javadoc);
    log.p("_Reverse Strategy Javadoc_: " + revFactory_javadoc);
    log.p("_Optimization Strategy Javadoc_: " + optimizationStrategy_javadoc);
  }

  protected abstract void intro(NotebookOutput log);
}
