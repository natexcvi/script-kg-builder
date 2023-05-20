package main

import (
	"fmt"
	"os"
	"path"
	"strings"

	kg "github.com/natexcvi/script-kg-builder/knowledge_graph"

	log "github.com/sirupsen/logrus"
)

func main() {
	log.SetLevel(log.DebugLevel)
	graph := &kg.KnowledgeGraph{
		Edges: []*kg.KGEdge{},
	}
	script, err := kg.LoadScript("scripts/12_years_a_slave")
	if err != nil {
		log.Fatalf("could not load script: %v", err)
	}
	for _, scene := range script.Scenes {
		newEdges, err := kg.LearnNewEdges(graph, scene)
		if err != nil {
			log.Fatalf("could not learn new edges: %v", err)
		}
		graph.Edges = append(graph.Edges, newEdges...)
	}
	fmt.Println(graph.Encode())
	if _, err := os.Stat("results"); os.IsNotExist(err) {
		if err := os.Mkdir("results", 0755); err != nil {
			log.Fatalf("could not create results directory: %v", err)
		}
	}
	if err := os.WriteFile(path.Join("results", "12_years_a_slave.txt"), []byte(strings.Trim(graph.Encode(), "\n")), 0644); err != nil {
		log.Fatalf("could not write graph encoding: %v", err)
	}
}
