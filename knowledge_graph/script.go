package knowledgegraph

import (
	"os"
	"path"
)

type Scene struct {
	Text string
}

type Script struct {
	Scenes []*Scene
}

func LoadScript(scriptDir string) (*Script, error) {
	files, err := os.ReadDir(scriptDir)
	if err != nil {
		return nil, err
	}
	// read all scenes from files
	scenes := []*Scene{}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		sceneText, err := os.ReadFile(path.Join(scriptDir, file.Name()))
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
