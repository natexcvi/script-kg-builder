package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path"
	"strings"

	"github.com/natexcvi/go-llm/engines"
	kg "github.com/natexcvi/script-kg-builder/knowledge_graph"
	"github.com/samber/lo"

	"github.com/schollz/progressbar/v3"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	totalTokenLimit   int
	mermaidOutputFile string
	shallow           bool
	model             string
	verbose           bool
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

var embedCmd = &cobra.Command{
	Use:   "embed kg_file output_csv",
	Short: "embed is a tool for embedding relations in a knowledge graph",
	Long: `embed is a tool for embedding relations in a knowledge graph.
The kg_file should contain a knowledge graph in the following format:
(head, relation, tail)
(head, relation, tail)
...
where each line represents a single edge in the knowledge graph.`,
	Run: func(cmd *cobra.Command, args []string) {
		apiToken := os.Getenv("OPENAI_API_KEY")
		if apiToken == "" {
			log.Fatal("OPENAI_API_KEY environment variable must be set")
		}
		embedder := kg.NewRelationEmbedder(http.DefaultClient, apiToken)
		kgFile := args[0]
		f, err := os.Open(kgFile)
		if err != nil {
			log.Fatalf("could not open kg file: %v", err)
		}
		defer f.Close()
		numLines, err := numLinesInFile(kgFile)
		outputFilePath := args[1]
		outputFile, err := os.OpenFile(outputFilePath, os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatalf("could not open output file: %v", err)
		}
		defer outputFile.Close()
		if err := writeEmbeddingsFileHeader(outputFile); err != nil {
			log.Fatalf("could not write embeddings file header: %v", err)
		}

		scn := bufio.NewScanner(f)
		pbar := progressbar.Default(int64(numLines), "embedding relations")
		embeddingsMap := make(map[string][]float64)
		edges := []*kg.KGEdge{}
		for scn.Scan() {
			pbar.Describe("parsing kg edge")
			curLine := scn.Text()
			if curLine == "---" {
				pbar.ChangeMax(pbar.GetMax() - 1)
				continue
			}
			edge, err := kg.ParseKGEdge(curLine)
			if err != nil {
				log.Errorf("could not parse kg edge: %v", err)
				pbar.ChangeMax(pbar.GetMax() - 1)
				continue
			}
			edges = append(edges, edge)
			pbar.Describe("embedding relation")
			if _, ok := embeddingsMap[edge.Relation]; ok {
				pbar.Add(1)
				continue
			}
			embeddings, err := embedder.EmbedRelation(edge.Relation)
			if err != nil {
				log.Fatalf("could not embed relation: %v", err)
			}
			embeddingsMap[edge.Relation] = embeddings
			pbar.Add(1)
		}
		pbar.Finish()
		if err := writeEmbeddingsToFile(edges, embeddingsMap, outputFile); err != nil {
			log.Fatalf("could not write embeddings to file: %v", err)
		}
		log.Infof("wrote embeddings to file %q", outputFilePath)
		log.Infof("total tokens used: %d", embedder.TokensUsed())
	},
	Args: cobra.ExactArgs(2),
}

func numLinesInFile(filePath string) (int, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return 0, fmt.Errorf("could not open file %q: %w", filePath, err)
	}
	defer f.Close()
	scn := bufio.NewScanner(f)
	numLines := 0
	for scn.Scan() {
		numLines++
	}
	return numLines, nil
}

func writeEmbeddingsFileHeader(outputFile *os.File) (err error) {
	header := "head,relation,relation_embedding,tail\n"
	totalWritten := 0
	for {
		var n int
		if n, err = outputFile.WriteString(header); err != nil {
			return fmt.Errorf("could not write header to file: %w", err)
		}
		totalWritten += n
		if totalWritten == len(header) {
			break
		}
	}
	return nil
}

func writeEmbeddingsToFile(edges []*kg.KGEdge, embeddingsMap map[string][]float64, outputFile *os.File) (err error) {
	outputWriter := csv.NewWriter(outputFile)
	defer outputWriter.Flush()
	for _, edge := range edges {
		marshaledEmbeddings, err := json.Marshal(embeddingsMap[edge.Relation])
		if err != nil {
			return fmt.Errorf("could not marshal embeddings: %w", err)
		}
		if err := outputWriter.Write([]string{
			edge.Head,
			edge.Relation,
			string(marshaledEmbeddings),
			edge.Tail,
		}); err != nil {
			return fmt.Errorf("could not write embeddings to file: %w", err)
		}
	}
	return nil
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

func writeSceneGraph(edges []*kg.KGEdge, outputFile string) error {
	f, err := os.Create(outputFile)
	if err != nil {
		return fmt.Errorf("could not create file %q: %w", outputFile, err)
	}
	defer f.Close()
	if _, err := f.WriteString(kg.NewMermaidGraph(edges).String()); err != nil {
		return fmt.Errorf("could not write mermaid graph to file %q: %w", outputFile, err)
	}
	return nil
}

func createIfNotExists(dir string) error {
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	return os.Mkdir(dir, 0755)
}

func runner(scriptDir, outputFile string) {
	logLevel := lo.If(verbose, log.DebugLevel).Else(log.InfoLevel)
	log.SetLevel(logLevel)
	graph := &kg.KnowledgeGraph{
		Edges: []*kg.KGEdge{},
	}
	script, err := kg.LoadScript(scriptDir)
	if err != nil {
		log.Fatalf("could not load script: %v", err)
	}
	pbar := progressbar.Default(int64(len(script.Scenes)))
	engine := engines.NewGPTEngine(os.Getenv("OPENAI_API_KEY"), model).WithTotalTokenLimit(totalTokenLimit)
	agent := kg.NewKGBuilderAgent(engine)
	for i, scene := range script.Scenes {
		newEdges, err := agent.LearnNewEdges(graph, scene, lo.If(shallow, kg.KGBuildModeShallow).Else(kg.KGBuildModeDeep))
		if err != nil {
			log.Fatalf("could not learn new edges: %v", err)
		}
		graph.Edges = append(graph.Edges, newEdges...)
		if err := appendSceneEdgesToFile(newEdges, outputFile); err != nil {
			log.Fatalf("could not append scene edges to file: %v", err)
		}
		if mermaidOutputFile != "" {
			graphsDir := strings.TrimSuffix(mermaidOutputFile, path.Ext(mermaidOutputFile))
			if err := createIfNotExists(graphsDir); err != nil {
				log.Fatalf("could not create graphs directory: %v", err)
			}
			graphsFile := fmt.Sprintf("%s/%d.mmd", graphsDir, i)
			if err := writeSceneGraph(newEdges, graphsFile); err != nil {
				log.Fatalf("could not append scene graph to mermaid file: %v", err)
			}
		}
		pbar.Add(1)
	}
	if mermaidOutputFile != "" {
		if err := os.WriteFile(mermaidOutputFile, []byte(kg.NewMermaidGraph(graph.Edges).String()), 0644); err != nil {
			log.Fatalf("could not write mermaid graph: %v", err)
		}
	}
}

func init() {
	rootCmd.PersistentFlags().IntVarP(&totalTokenLimit, "total-token-limit", "l", 10000, "total token limit for OpenAI engine")
	rootCmd.PersistentFlags().StringVarP(&mermaidOutputFile, "mermaid-output-file", "m", "", "file to write mermaid graph to")
	rootCmd.PersistentFlags().BoolVarP(&shallow, "shallow", "s", false, "build knowledge graph in shallow mode")
	rootCmd.PersistentFlags().StringVarP(&model, "model", "M", "gpt-3.5-turbo", "OpenAI chat completion model")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")
	rootCmd.AddCommand(embedCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("could not execute root command: %v", err)
	}
}
