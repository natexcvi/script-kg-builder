package knowledgegraph

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/hashicorp/go-multierror"
	"github.com/natexcvi/go-llm/engines"
	"github.com/samber/lo"
	log "github.com/sirupsen/logrus"
)

type KGBuildMode string

const (
	KGBuildModeDeep    KGBuildMode = "deep"
	KGBuildModeShallow KGBuildMode = "shallow"
)

type kgBuilderAgent struct {
	engine engines.LLMWithFunctionCalls
}

func (a *kgBuilderAgent) modeDependentInstruction(mode KGBuildMode) string {

	switch mode {
	case KGBuildModeShallow:
		return "The task involves learning knowledge graphs " +
			"from screenplays. Add new relations to the knowledge graph based" +
			"on the content of the scene I'll give you. " +
			"Your output should be a list of triplet edges " +
			"of the form `(head, relation, tail)`. Focus on " +
			"local, scene-specific relations. Answer right away without " +
			"much reflection."
	default:
		return "The task involves learning knowledge graphs " +
			"from screenplays. You are given a knowledge graph and a new " +
			"scene from the screenplay. You should derive new relations " +
			"from the scene and add them to the knowledge graph. You can " +
			"assume that the knowledge graph is a list of triplets of the " +
			"form `(head, relation, tail)`, where all three are short " +
			"strings that do not contain commas. Try to use existing relation " +
			"kinds if appropriate, and use a new kind of relation only if you " +
			"can't find an existing one that fits the relation you detect. " +
			"Make sure you don't add duplicate edges to the knowledge graph. " +
			"Add relations of the deep kind: relations that represent plot-wide facts (that are not trivial). " +
			"Your output should be a list of new triplets to add to the " +
			"knowledge graph. "
	}
}

func (a *kgBuilderAgent) kgParserFromFunctionCall(s string) (kg *KnowledgeGraph, err error) {
	var parsedArgs struct {
		Edges []*KGEdge `json:"edges"`
	}
	if err := json.Unmarshal([]byte(s), &parsedArgs); err != nil {
		return nil, err
	}
	return &KnowledgeGraph{
		Edges: parsedArgs.Edges,
	}, nil
}

func (a *kgBuilderAgent) kgParser(s string) (kg *KnowledgeGraph, err error) {
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
}

func (a *kgBuilderAgent) LearnNewEdges(existingKG *KnowledgeGraph, scene *Scene, mode KGBuildMode) ([]*KGEdge, error) {
	instruction := a.modeDependentInstruction(mode)
	examples := []*engines.ChatMessage{
		{
			Role: engines.ConvRoleSystem,
			Text: instruction,
		},
		{
			Role: engines.ConvRoleUser,
			Text: "Here is the scene:\n\n" + "INT. HOUSE/KITCHEN - MORNING\nMary leans in and kisses Tim.",
		},
		{
			Role: engines.ConvRoleAssistant,
			FunctionCall: &engines.FunctionCall{
				Name: "add_edges",
				Args: `{"edges": [{"head": "Mary", "relation": "kisses", "tail": "Tim"}, {"head": "Mary", "relation": "romantically involved with", "tail": "Tim"}]}`,
			},
		},
		{
			Role: engines.ConvRoleUser,
			Text: "Here is the scene:\n\n" + "INT. CUPACOFFEE/COUNTER - AFTERNOON\nMary is making coffee for a customer.",
		},
		{
			Role: engines.ConvRoleAssistant,
			FunctionCall: &engines.FunctionCall{
				Name: "add_edges",
				Args: `{"edges": [{"head": "Mary", "relation": "works at", "tail": "CupACoffee"}]}`,
			},
		},
	}
	input := &engines.ChatMessage{
		Role: engines.ConvRoleUser,
		Text: "Here is the scene:\n\n" + scene.Text,
	}
	prompt := &engines.ChatPrompt{
		History: append(examples, input),
	}
	log.Debugf("Prompt: %s", strings.Join(lo.Map(prompt.History, func(m *engines.ChatMessage, _ int) string {
		if m.FunctionCall != nil {
			return fmt.Sprintf("%s: %s(%s)", m.Role, m.FunctionCall.Name, m.FunctionCall.Args)
		}
		return fmt.Sprintf("%s: %s", m.Role, m.Text)
	}), "\n"))

	var finalErr *multierror.Error
	for attempt := 0; attempt < 5; attempt++ {
		response, err := a.engine.PredictWithFunctions(prompt)
		if err != nil {
			finalErr = multierror.Append(finalErr, fmt.Errorf("prediction failed: %w", err))
			log.Errorf("prediction failed: %v", err)
			continue
		}
		var newGraph *KnowledgeGraph
		if response.FunctionCall != nil {
			newGraph, err = a.kgParserFromFunctionCall(response.FunctionCall.Args)
		} else {
			newGraph, err = a.kgParser(response.Text)
		}
		if err != nil {
			finalErr = multierror.Append(finalErr, fmt.Errorf("could not parse response: %w", err))
			log.Errorf("could not parse response: %s", err)
			continue
		}
		return newGraph.Edges, nil
	}
	return nil, finalErr.ErrorOrNil()
}

func NewKGBuilderAgent(engine engines.LLMWithFunctionCalls) *kgBuilderAgent {
	engine.SetFunctions(engines.FunctionSpecs{
		Name:        "add_edges",
		Description: "adds new edges to the knowledge graph",
		Parameters: &engines.ParameterSpecs{
			Type:        "object",
			Description: "",
			Properties: map[string]*engines.ParameterSpecs{
				"edges": {
					Type:        "array",
					Description: "a list of triplet edges",
					Items: &engines.ParameterSpecs{
						Type:        "object",
						Description: "a triplet edge",
						Properties: map[string]*engines.ParameterSpecs{
							"head": {
								Type:        "string",
								Description: "the source entity",
							},
							"relation": {
								Type:        "string",
								Description: "the relation kind",
							},
							"tail": {
								Type:        "string",
								Description: "the target entity",
							},
						},
					},
				},
			},
		},
	})
	return &kgBuilderAgent{
		engine: engine,
	}
}
