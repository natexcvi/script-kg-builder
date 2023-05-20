package main

import (
	"fmt"
	"os"
	"path"
	"strings"

	kg "github.com/natexcvi/script-kg-builder/knowledge_graph"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	totalTokenLimit int
)

var rootCmd = &cobra.Command{
	Use:   "script-kg-builder SCRIPT_DIR",
	Short: "script-kg-builder is a tool for building knowledge graphs from movie scripts",
	Run: func(cmd *cobra.Command, args []string) {
		if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			log.Fatal("OPENAI_API_KEY environment variable must be set")
		}
		runner()
	},
	Args: cobra.ExactArgs(1),
}

func init() {
	rootCmd.PersistentFlags().IntVarP(&totalTokenLimit, "total-token-limit", "l", 10000, "total token limit for OpenAI engine")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("could not execute root command: %v", err)
	}
}

func runner() {
	log.SetLevel(log.DebugLevel)
	graph := &kg.KnowledgeGraph{
		Edges: []*kg.KGEdge{},
	}
	script, err := kg.LoadScript("scripts/12_years_a_slave")
	if err != nil {
		log.Fatalf("could not load script: %v", err)
	}
	for _, scene := range script.Scenes {
		newEdges, err := kg.LearnNewEdges(graph, scene, totalTokenLimit)
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
