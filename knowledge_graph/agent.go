package knowledgegraph

import (
	"fmt"
	"os"
	"strings"

	"github.com/natexcvi/go-llm/agents"
	"github.com/natexcvi/go-llm/engines"
	"github.com/natexcvi/go-llm/memory"
)

type KGBuilderInput struct {
	Scene      *Scene
	ExistingKG *KnowledgeGraph
}

func (i *KGBuilderInput) Encode() string {
	return fmt.Sprintf("Here is the current state of the knowledge graph:\n%s\n\nAnd here is a new scene from the film:\n%s", i.ExistingKG.Encode(), i.Scene.Text)
}

func (i *KGBuilderInput) Schema() string {
	return "<representation of the already learned knowledge graph>\n<text of the current scene>"
}

func LearnNewEdges(existingKG *KnowledgeGraph, scene *Scene) ([]*KGEdge, error) {
	engine := engines.NewGPTEngine(os.Getenv("OPENAI_API_KEY"), "gpt-3.5-turbo")
	task := &agents.Task[*KGBuilderInput, *KnowledgeGraph]{
		Description: "The task involves learning knowledge graphs " +
			"from screenplays. You are given a knowledge graph and a new " +
			"scene from the screenplay. You should derive new relations " +
			"from the scene and add them to the knowledge graph. You can " +
			"assume that the knowledge graph is a list of triplets of the " +
			"form (head, relation, tail), where all three are short " +
			"strings that do not contain commas. Try to use existing relation " +
			"kinds if appropriate, and use a new kind of relation only if you " +
			"can't find an existing one that fits the relation you detect. " +
			"Your output should be a list of new triplets to add to the " +
			"knowledge graph. ",
		Examples: []agents.Example[*KGBuilderInput, *KnowledgeGraph]{
			{
				Input: &KGBuilderInput{
					Scene: &Scene{
						Text: "INT. HOUSE/KITCHEN - MORNING\nMary leans in and kisses Tim.",
					},
					ExistingKG: &KnowledgeGraph{
						Edges: []*KGEdge{},
					},
				},
				Answer: &KnowledgeGraph{
					Edges: []*KGEdge{
						{
							Head:     "Mary",
							Relation: "romantically involved with",
							Tail:     "Tim",
						},
					},
				},
			},
			{
				Input: &KGBuilderInput{
					Scene: &Scene{
						Text: "INT. CUPACOFFEE/COUNTER - AFTERNOON\nMary is making coffee for a customer.",
					},
					ExistingKG: &KnowledgeGraph{
						Edges: []*KGEdge{
							{
								Head:     "Mary",
								Relation: "romantically involved with",
								Tail:     "Tim",
							},
						},
					},
				},
				Answer: &KnowledgeGraph{
					Edges: []*KGEdge{
						{
							Head:     "Mary",
							Relation: "works at",
							Tail:     "Cafe",
						},
					},
				},
			},
		},
		AnswerParser: func(s string) (kg *KnowledgeGraph, err error) {
			edges := []*KGEdge{}
			for _, line := range strings.Split(s, "\n") {
				if line == "" {
					continue
				}
				edge, err := ParseKGEdge(line)
				if err != nil {
					return nil, err
				}
				edges = append(edges, edge)
			}
			return &KnowledgeGraph{
				Edges: edges,
			}, nil
		},
	}
	agent := agents.NewChainAgent(engine, task, memory.NewBufferedMemory(10)).WithMaxSolutionAttempts(5).WithRestarts(3)
	newGraph, err := agent.Run(&KGBuilderInput{
		Scene:      scene,
		ExistingKG: &KnowledgeGraph{},
	})
	if err != nil {
		return nil, err
	}
	return newGraph.Edges, nil
}
