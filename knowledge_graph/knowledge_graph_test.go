package knowledgegraph

import (
	"testing"
)

func Test_KnowledgeGraph_Encode(t *testing.T) {
	graph := &KnowledgeGraph{}

	cases := []struct {
		name  string
		edges []*KGEdge
		want  string
	}{
		{"Empty Graph", []*KGEdge{}, "\n<empty knowledge graph>"},
		{"Non-Empty Graph 1", []*KGEdge{
			{Head: "A", Relation: "REL", Tail: "B"},
			{Head: "B", Relation: "REL", Tail: "C"},
			{Head: "C", Relation: "REL", Tail: "D"},
		}, "\n(A, REL, B)\n(B, REL, C)\n(C, REL, D)"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			graph.Edges = c.edges
			if got := graph.Encode(); got != c.want {
				t.Errorf("Encode() returned %v, want %v", got, c.want)
			}
		})
	}
}

func Test_KnowledgeGraph_Schema(t *testing.T) {
	graph := &KnowledgeGraph{}

	cases := []struct {
		name string
		want string
	}{
		{"Schema Test 1", "<line-separated list of triplets of the form (head, relation, tail)>"},
		{"Schema Test 2", "<line-separated list of triplets of the form (head, relation, tail)>"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := graph.Schema(); got != c.want {
				t.Errorf("Schema() returned %v, want %v", got, c.want)
			}
		})
	}
}

func Test_ParseKGEdge(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		wantEdge KGEdge
		wantErr  bool
	}{
		{"Valid Edge", "(A, REL, B)", KGEdge{Head: "A", Relation: "REL", Tail: "B"}, false},
		{"Invalid Edge", "(A, REL, )", KGEdge{}, true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			edge, err := ParseKGEdge(c.input)
			if err != nil {
				if c.wantErr {
					return
				}
				t.Fatalf("Unexpected error when parsing %q: %v", c.input, err)
			}
			if *edge != c.wantEdge {
				t.Errorf("Parsed edge %v, want %v", *edge, c.wantEdge)
			}
		})
	}
}
