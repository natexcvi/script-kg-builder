package knowledgegraph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadScript(t *testing.T) {
	tests := []struct {
		name                   string
		scriptPath             string
		expNumScenes           int
		expFirstSceneToContain string
	}{
		{
			name:                   "sanity",
			scriptPath:             "../scripts/12_years_a_slave",
			expNumScenes:           3,
			expFirstSceneToContain: "INT. HOUSE/LIVING ROOM - EVENING",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			script, err := LoadScript(tt.scriptPath)
			if err != nil {
				t.Errorf("LoadScript() error = %v", err)
			}

			require.Equal(t, len(script.Scenes), tt.expNumScenes)

			assert.Contains(t, script.Scenes[0].Text, tt.expFirstSceneToContain)
		})
	}
}
