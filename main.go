package main

import (
	"fmt"
	"os"

	kg "github.com/natexcvi/script-kg-builder/knowledge_graph"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	totalTokenLimit   int
	mermaidOutputFile string
)

var rootCmd = &cobra.Command{
	Use:   "script-kg-builder script_dir output_file",
	Short: "script-kg-builder is a tool for building knowledge graphs from movie scripts",
	Long: `script-kg-builder is a tool for building knowledge graphs from movie scripts.
The script directory should contain a set of files named <scene number>.txt, where
<scene number> is the chronological order of the scene in the movie. For example,
the first scene in the movie should be named 1.txt, the second scene should be named
2.txt, and so on. The script directory should not contain any other files.`,
	Run: func(cmd *cobra.Command, args []string) {
		if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			log.Fatal("OPENAI_API_KEY environment variable must be set")
		}
		scriptDir := args[0]
		if err := validateDirectory(scriptDir); err != nil {
			log.Fatalf("invalid script directory: %v", err)
		}
		outputFile := args[1]
		if err := validateOutputFilePath(outputFile); err != nil {
			log.Fatalf("invalid output file path: %v", err)
		}
		runner(scriptDir, outputFile)
	},
	Args: cobra.ExactArgs(2),
}

func validateDirectory(dir string) error {
	fInfo, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !fInfo.IsDir() {
		return fmt.Errorf("%q is not a directory", dir)
	}
	return nil
}

func validateOutputFilePath(filePath string) error {
	if _, err := os.Stat(filePath); err == nil {
		return fmt.Errorf("%q already exists", filePath)
	}
	if _, err := os.Create(filePath); err != nil {
		return fmt.Errorf("could not create file %q: %w", filePath, err)
	}
	return nil
}

func appendSceneEdgesToFile(edges []*kg.KGEdge, outputFile string) error {
	f, err := os.OpenFile(outputFile, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("could not open file %q: %w", outputFile, err)
	}
	defer f.Close()
	for _, edge := range edges {
		if _, err := f.WriteString(fmt.Sprintf("%s\n", edge.String())); err != nil {
			return fmt.Errorf("could not write edge to file %q: %w", outputFile, err)
		}
	}
	f.WriteString("---\n")
	return nil
}

func runner(scriptDir, outputFile string) {
	log.SetLevel(log.DebugLevel)
	graph := &kg.KnowledgeGraph{
		Edges: []*kg.KGEdge{},
	}
	script, err := kg.LoadScript(scriptDir)
	if err != nil {
		log.Fatalf("could not load script: %v", err)
	}
	for _, scene := range script.Scenes {
		newEdges, err := kg.LearnNewEdges(graph, scene, totalTokenLimit)
		if err != nil {
			log.Fatalf("could not learn new edges: %v", err)
		}
		graph.Edges = append(graph.Edges, newEdges...)
		if err := appendSceneEdgesToFile(newEdges, outputFile); err != nil {
			log.Fatalf("could not append scene edges to file: %v", err)
		}
	}
	if mermaidOutputFile != "" {
		if err := os.WriteFile(mermaidOutputFile, []byte(kg.NewMermaidGraph(graph).String()), 0644); err != nil {
			log.Fatalf("could not write mermaid graph: %v", err)
		}
	}
}

func init() {
	rootCmd.PersistentFlags().IntVarP(&totalTokenLimit, "total-token-limit", "l", 10000, "total token limit for OpenAI engine")
	rootCmd.PersistentFlags().StringVarP(&mermaidOutputFile, "mermaid-output-file", "m", "", "file to write mermaid graph to")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("could not execute root command: %v", err)
	}
}
