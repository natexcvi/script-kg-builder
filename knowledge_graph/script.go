package knowledgegraph

import (
	"fmt"
	"os"
	"path"
	"regexp"
	"strconv"

	"golang.org/x/exp/slices"
)

var errNotASceneFile = fmt.Errorf("not a scene file")

type Scene struct {
	Text string
}

type Script struct {
	Scenes []*Scene
}

func parseSceneFileName(fileName string) (int, error) {
	sceneFileNameRegex := regexp.MustCompile(`^([0-9]+)\.txt$`)
	matches := sceneFileNameRegex.FindStringSubmatch(fileName)
	if len(matches) != 2 {
		return 0, errNotASceneFile
	}
	return strconv.Atoi(matches[1])
}

type sceneFile struct {
	index    int
	filename string
}

func LoadScript(scriptDir string) (*Script, error) {
	files, err := os.ReadDir(scriptDir)
	if err != nil {
		return nil, err
	}
	// read all scenes from files
	scenes := []*Scene{}
	sceneFiles := make([]sceneFile, 0)
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		sceneIndex, err := parseSceneFileName(file.Name())
		if err == errNotASceneFile {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("could not parse scene file name: %w", err)
		}
		sceneFiles = append(sceneFiles, sceneFile{
			index:    sceneIndex,
			filename: file.Name(),
		})
	}

	slices.SortFunc(sceneFiles, func(a, b sceneFile) bool {
		return a.index < b.index
	})

	for _, sceneFile := range sceneFiles {
		sceneText, err := os.ReadFile(path.Join(scriptDir, sceneFile.filename))
		if err != nil {
			return nil, err
		}
		scenes = append(scenes, &Scene{
			Text: string(sceneText),
		})
	}
	return &Script{
		Scenes: scenes,
	}, nil
}
