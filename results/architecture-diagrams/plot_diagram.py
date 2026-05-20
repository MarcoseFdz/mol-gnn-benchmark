import argparse
import graphviz

def generate_diagram(model_type, hidden_dim, output_name):
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", size="12,8", dpi="300")
    dot.attr("node", fontname="Helvetica", shape="box", style="filled,rounded", color="#1A252C", fillcolor="#F2F4F7", penwidth="2", fontsize="11")
    dot.attr("edge", fontname="Helvetica", color="#4A5568", penwidth="1.5", arrowhead="vee", arrowsize="0.8", fontsize="9")

    dot.node("features", "Node Features\nShape: (B, N, 51)", fillcolor="#E3F2FD", color="#1E88E5")
    dot.node("adjacency", "Adjacency Matrix\nShape: (B, N, N)", fillcolor="#FFF3E0", color="#FB8C00")

    current = "features"

    if model_type.lower() == "gat":
        dot.node("projection", "Input Projection\nDense (D = " + str(hidden_dim) + ")", fillcolor="#EDE7F6", color="#5E35B1")
        dot.edge("features", "projection")
        current = "projection"

    for i in range(1, 4):
        cluster_name = "cluster_block_" + str(i)
        with dot.subgraph(name=cluster_name) as c:
            if model_type.lower() == "gcn":
                c.attr(label="GCN Block " + str(i), fontname="Helvetica-Bold", fontsize="12", style="dashed", color="#A0AEC0", bgcolor="#FAFAFA")
                layer_node = "gcn_" + str(i)
                norm_node = "ln_" + str(i)
                c.node(layer_node, "GCN Layer " + str(i) + "\nUnits: " + str(hidden_dim), fillcolor="#E8F5E9", color="#43A047")
                c.node(norm_node, "Layer Norm " + str(i), fillcolor="#F5F5F5", color="#9E9E9E")
                c.edge(layer_node, norm_node)
                dot.edge(current, layer_node)
                dot.edge("adjacency", layer_node, style="dashed", constraint="false")
                current = norm_node
            elif model_type.lower() == "sage":
                c.attr(label="GraphSAGE Block " + str(i), fontname="Helvetica-Bold", fontsize="12", style="dashed", color="#A0AEC0", bgcolor="#FAFAFA")
                layer_node = "sage_" + str(i)
                norm_node = "ln_" + str(i)
                c.node(layer_node, "SAGELayer " + str(i) + "\nUnits: " + str(hidden_dim), fillcolor="#E8F5E9", color="#43A047")
                c.node(norm_node, "Layer Norm " + str(i), fillcolor="#F5F5F5", color="#9E9E9E")
                c.edge(layer_node, norm_node)
                dot.edge(current, layer_node)
                dot.edge("adjacency", layer_node, style="dashed", constraint="false")
                current = norm_node
            elif model_type.lower() == "gat":
                c.attr(label="Graph Transformer Block " + str(i), fontname="Helvetica-Bold", fontsize="12", style="dashed", color="#A0AEC0", bgcolor="#FAFAFA")
                gnn = "gat_" + str(i)
                ffn = "ffn_" + str(i)
                c.node(gnn, "GAT Layer " + str(i) + "\nHeads: 8, Units: 16", fillcolor="#E8F5E9", color="#43A047")
                c.node(ffn, "Feed-Forward " + str(i) + "\nDense (D = " + str(hidden_dim) + ")", fillcolor="#FCE4EC", color="#D81B60")
                c.edge(gnn, ffn)
                dot.edge(current, gnn)
                dot.edge("adjacency", gnn, style="dashed", constraint="false")
                dot.edge(current, ffn, style="dotted", label="Residual")
                current = ffn

    with dot.subgraph(name="cluster_readout") as c:
        c.attr(label="Readout Phase", fontname="Helvetica-Bold", fontsize="12", style="dashed", color="#A0AEC0")
        c.node("mean", "Global Mean Pool", fillcolor="#E0F7FA", color="#00ACC1")
        c.node("max", "Global Max Pool", fillcolor="#E0F7FA", color="#00ACC1")
        c.node("sum", "Global Sum Pool", fillcolor="#E0F7FA", color="#00ACC1")
        c.node("concat", "Concatenate\nShape: (B, 3D)", fillcolor="#E0F2F1", color="#00897B")
        c.edge("mean", "concat")
        c.edge("max", "concat")
        c.edge("sum", "concat")

    dot.edge(current, "mean")
    dot.edge(current, "max")
    dot.edge(current, "sum")

    dot.node("mlp", "MLP Head\nDense Layers", fillcolor="#FFFDE7", color="#FDD835")
    dot.node("output", "Output Layer\nShape: (B, Classes)", fillcolor="#E8F5E9", color="#43A047")

    dot.edge("concat", "mlp")
    dot.edge("mlp", "output")

    dot.render(output_name, cleanup=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, choices=["gcn", "sage", "gat"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="gnn_architecture")
    args = parser.parse_args()
    generate_diagram(args.type, args.hidden_dim, args.output)
