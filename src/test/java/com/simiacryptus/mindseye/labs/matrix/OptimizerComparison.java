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
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public abstract class OptimizerComparison extends NotebookReportBase {

  protected ImageProblemData data;
  protected FwdNetworkFactory fwdFactory;
  protected RevNetworkFactory revFactory;
  protected int timeoutMinutes = 10;

  public OptimizerComparison(final FwdNetworkFactory fwdFactory, final RevNetworkFactory revFactory,
                             final ImageProblemData data) {
    this.fwdFactory = fwdFactory;
    this.revFactory = revFactory;
    this.data = data;
  }

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Experiments;
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return OptimizerComparison.class;
  }

  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }

  @Nonnull
  public OptimizerComparison setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  @Test
  @Category(TestCategories.Report.class)
  public void classification() {
    run(log -> classification(log), getClass().getSimpleName(), "Classification");
  }

  public void classification(@Nonnull NotebookOutput log) {
    compare(log, opt -> {
      return new ClassifyProblem(fwdFactory, opt, data, 10).setTimeoutMinutes(timeoutMinutes).run(log).getHistory();
    });
  }

  public abstract void compare(NotebookOutput log, Function<OptimizationStrategy, List<StepRecord>> test);

  @Test
  @Category(TestCategories.Report.class)
  public void encoding() {
    run(log -> encoding(log), getClass().getSimpleName(), "Encoding");
  }

  public void encoding(@Nonnull NotebookOutput log) {
    compare(log, opt -> {
      return new EncodingProblem(revFactory, opt, data, 20).setTimeoutMinutes(timeoutMinutes).setTrainingSize(1000)
          .run(log).getHistory();
    });
  }

}
