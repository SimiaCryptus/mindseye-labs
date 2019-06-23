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

package com.simiacryptus.mindseye.network.util;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;

@SuppressWarnings("serial")
public class PolynomialNetwork extends DAGNetwork {

  protected final int[] inputDims;
  protected final int[] outputDims;
  @Nullable
  protected Layer alpha = null;
  @Nullable
  protected Layer alphaBias = null;
  @Nonnull
  protected List<Correcton> corrections = new ArrayList<>();
  protected DAGNode head;

  public PolynomialNetwork(final int[] inputDims, final int[] outputDims) {
    super(1);
    this.inputDims = inputDims;
    this.outputDims = outputDims;
  }


  protected PolynomialNetwork(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    head = getNodeById(UUID.fromString(json.get("head").getAsString()));
    Map<UUID, Layer> layersById = getLayersById();
    if (json.get("alpha") != null) {
      alpha = layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    }
    if (json.get("alphaBias") != null) {
      alphaBias = layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    }
    inputDims = PolynomialNetwork.toIntArray(json.getAsJsonArray("inputDims"));
    outputDims = PolynomialNetwork.toIntArray(json.getAsJsonArray("outputDims"));
    json.getAsJsonArray("corrections").forEach(item -> {
      corrections.add(new Correcton(item.getAsJsonObject()));
    });
  }

  public static PolynomialNetwork fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PolynomialNetwork(json, rs);
  }

  @Nonnull
  public static int[] toIntArray(@Nonnull final JsonArray dims) {
    @Nonnull final int[] x = new int[dims.size()];
    int j = 0;
    for (@Nonnull final Iterator<JsonElement> i = dims.iterator(); i.hasNext(); ) {
      x[j++] = i.next().getAsInt();
    }
    return x;
  }

  @Nonnull
  public static JsonArray toJson(@Nonnull final int[] dims) {
    @Nonnull final JsonArray array = new JsonArray();
    for (final int i : dims) {
      array.add(new JsonPrimitive(i));
    }
    return array;
  }

  @Override
  protected void _free() {
    head.freeRef();
    alpha.freeRef();
    alphaBias.freeRef();
    super._free();
  }

  public void addTerm(final double power) {
    corrections.add(new Correcton(power,
        newBias(outputDims, 1.0),
        newSynapse(0.0)
    ));
  }

  @Override
  public synchronized DAGNode getHead() {
    if (null == head) {
      synchronized (this) {
        if (null == head) {
          if (null == alpha) {
            alpha = newSynapse(1e-8);
            alphaBias = newBias(inputDims, 0.0);
          }
          reset();
          final DAGNode input = getInput(0);
          @Nonnull final ArrayList<DAGNode> terms = new ArrayList<>();
          terms.add(add(alpha, add(alphaBias, input)));
          for (@Nonnull final Correcton c : corrections) {
            terms.add(c.add(input));
          }
          head = terms.size() == 1 ? terms.get(0) : add(newProductLayer(), terms.toArray(new DAGNode[]{}));
        }
      }
    }
    head.addRef();
    return head;
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    assertConsistent();
    @Nullable final UUID head = getHeadId();
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("head", head.toString());
    if (null != alpha) {
      json.addProperty("alpha", alpha.getId().toString());
    }
    if (null != alphaBias) {
      json.addProperty("alphaBias", alpha.getId().toString());
    }
    json.add("inputDims", PolynomialNetwork.toJson(inputDims));
    json.add("outputDims", PolynomialNetwork.toJson(outputDims));
    @Nonnull final JsonArray elements = new JsonArray();
    for (@Nonnull final Correcton c : corrections) {
      elements.add(c.getJson());
    }
    json.add("corrections", elements);
    assert null != Layer.fromJson(json) : "Smoke apply deserialization";
    return json;
  }

  @Nonnull
  public Layer newBias(final int[] dims, final double weight) {
    return new BiasLayer(dims).setWeights(i -> weight);
  }

  @Nonnull
  public Layer newNthPowerLayer(final double power) {
    return new NthPowerActivationLayer().setPower(power);
  }

  @Nonnull
  public Layer newProductLayer() {
    return new ProductInputsLayer();
  }

  @Nonnull
  public Layer newSynapse(final double weight) {
    return new FullyConnectedLayer(inputDims, outputDims).set(() -> weight * (Math.random() - 1));
  }

  public class Correcton {
    public final Layer bias;
    public final Layer factor;
    public final double power;

    public Correcton(final double power, final Layer bias, final Layer factor) {
      this.power = power;
      this.bias = bias;
      this.factor = factor;
    }

    public Correcton(@Nonnull final JsonObject json) {
      power = json.get("power").getAsDouble();
      Map<UUID, Layer> layersById = getLayersById();
      bias = layersById.get(UUID.fromString(json.get("bias").getAsString()));
      factor = layersById.get(UUID.fromString(json.get("factor").getAsString()));
    }

    public DAGNode add(final DAGNode input) {
      return PolynomialNetwork.this.add(newNthPowerLayer(power), PolynomialNetwork.this.add(bias, PolynomialNetwork.this.add(factor, input)));
    }

    @Nonnull
    public JsonObject getJson() {
      @Nonnull final JsonObject json = new JsonObject();
      json.addProperty("bias", bias.getId().toString());
      json.addProperty("factor", factor.getId().toString());
      json.addProperty("power", power);
      return json;
    }
  }

}
