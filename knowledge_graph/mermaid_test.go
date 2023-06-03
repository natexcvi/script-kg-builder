package knowledgegraph

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMermaidGraphGeneration(t *testing.T) {
	testCases := []struct {
		name string
		kg   *KnowledgeGraph
		want string
	}{
		{
			name: "empty graph",
			kg:   &KnowledgeGraph{},
			want: "graph TD\n",
		},
		{
			name: "single edge",
			kg: &KnowledgeGraph{
				Edges: []*KGEdge{
					{
						Head:     "a",
						Relation: "b",
						Tail:     "c",
					},
				},
			},
			want: "graph TD\n\t0[a]\n\t1[c]\n\t0 --\"b\"--> 1\n",
		},
		{
			name: "multiple edges",
			kg: &KnowledgeGraph{
				Edges: []*KGEdge{
					{
						Head:     "a",
						Relation: "b",
						Tail:     "c",
					},
					{
						Head:     "c",
						Relation: "d",
						Tail:     "e",
					},
				},
			},
			want: "graph TD\n\t0[a]\n\t1[c]\n\t2[e]\n\t0 --\"b\"--> 1\n\t1 --\"d\"--> 2\n",
		},
		{
			name: "multiple edges with same vertices",
			kg: &KnowledgeGraph{
				Edges: []*KGEdge{
					{
						Head:     "a",
						Relation: "b",
						Tail:     "c",
					},
					{
						Head:     "c",
						Relation: "d",
						Tail:     "e",
					},
					{
						Head:     "a",
						Relation: "f",
						Tail:     "e",
					},
				},
			},
			want: "graph TD\n\t0[a]\n\t1[c]\n\t2[e]\n\t0 --\"b\"--> 1\n\t1 --\"d\"--> 2\n\t0 --\"f\"--> 2\n",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := NewMermaidGraph(tc.kg).String()
			assert.Equal(t, tc.want, got)
		})
	}
}
