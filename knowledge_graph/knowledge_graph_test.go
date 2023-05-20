package knowledgegraph

import (
	"testing"
)

func Test_KnowledgeGraph_Encode(t *testing.T) {
	// initialize empty graph
	graph := &KnowledgeGraph{}

	expected := "\n<empty knowledge graph>"
	if graph.Encode() != expected {
		t.Errorf("Encode() returned %q, expected %q", graph.Encode(), expected)
	}

	// add a few edges
	graph.Edges = []*KGEdge{
		{Head: "A", Relation: "REL", Tail: "B"},
		{Head: "B", Relation: "REL", Tail: "C"},
		{Head: "C", Relation: "REL", Tail: "D"},
	}

	expected = "\n(A, REL, B)\n(B, REL, C)\n(C, REL, D)"
	if graph.Encode() != expected {
		t.Errorf("Encode() returned %q, expected %q", graph.Encode(), expected)
	}
}

func Test_KnowledgeGraph_Schema(t *testing.T) {
	graph := &KnowledgeGraph{}
	expected := "<line-separated list of triplets of the form (head, relation, tail)>"
	if graph.Schema() != expected {
		t.Errorf("Schema() returned %q, expected %q", graph.Schema(), expected)
	}
}

func Test_ParseKGEdge(t *testing.T) {
	// test valid edge
	s := "(A, REL, B)"
	expected := &KGEdge{Head: "A", Relation: "REL", Tail: "B"}
	edge, err := ParseKGEdge(s)
	if err != nil {
		t.Errorf("Parsing failed with error: %s", err)
	}
	if *edge != *expected {
		t.Errorf("ParseKGEdge(%q) returned %q, expected %q", s, edge, expected)
	}

	// test invalid edge
	s = "(A, REL, )"
	edge, err = ParseKGEdge(s)
	if err == nil {
		t.Errorf("Expected parsing to fail, but it succeeded")
	}
}
