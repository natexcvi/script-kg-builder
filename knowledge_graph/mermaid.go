package knowledgegraph

import (
	"fmt"
	"strings"

	"github.com/samber/lo"
)

type mermaidGraph struct {
	repr string
}

func NewMermaidGraph(edges []*KGEdge) *mermaidGraph {
	vertices := lo.FlatMap(edges, func(edge *KGEdge, _ int) []string { return []string{edge.Head, edge.Tail} })
	vertices = lo.Uniq(vertices)
	mermaidRepr := strings.Builder{}
	mermaidRepr.WriteString("graph TD\n")
	for i, vertex := range vertices {
		mermaidRepr.WriteString(fmt.Sprintf("\t%d[%s]\n", i, vertex))
	}
	for _, edge := range edges {
		mermaidRepr.WriteString(fmt.Sprintf("\t%d --\"%s\"--> %d\n", lo.IndexOf(vertices, edge.Head), edge.Relation, lo.IndexOf(vertices, edge.Tail)))
	}
	return &mermaidGraph{mermaidRepr.String()}
}

func (mg *mermaidGraph) String() string {
	return mg.repr
}
