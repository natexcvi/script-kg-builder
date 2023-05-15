package knowledgegraph

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/samber/lo"
)

type KGEdge struct {
	Head     string
	Relation string
	Tail     string
}

func (e *KGEdge) String() string {
	return fmt.Sprintf("(%s, %s, %s)", e.Head, e.Relation, e.Tail)
}

type KnowledgeGraph struct {
	Edges []*KGEdge
}

func (g *KnowledgeGraph) Encode() string {
	if len(g.Edges) == 0 {
		return "\n<empty knowledge graph>"
	}
	return "\n" + strings.Join(lo.Map(g.Edges, func(e *KGEdge, _ int) string { return e.String() }), "\n")
}

func (g *KnowledgeGraph) Schema() string {
	return "<line-separated list of triplets of the form (head, relation, tail)>"
}

func ParseKGEdge(s string) (*KGEdge, error) {
	edgeExpr := regexp.MustCompile(`\(([^,]+), ([^,]+), ([^,]+)\)`)
	matches := edgeExpr.FindStringSubmatch(s)
	if matches == nil {
		return nil, fmt.Errorf("invalid edge: %q", s)
	}
	return &KGEdge{
		Head:     matches[1],
		Relation: matches[2],
		Tail:     matches[3],
	}, nil
}
