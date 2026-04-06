{{/*
Define the vllm-omni serve command from model + omniArgs.
If image.command is set, uses that as a full override.
*/}}
{{- define "chart.omni-command" -}}
{{- if .Values.image.command }}
{{-   toYaml .Values.image.command }}
{{- else }}
- "vllm"
- "serve"
- {{ .Values.model | quote }}
- "--omni"
- "--host"
- "0.0.0.0"
- "--port"
- {{ include "chart.container-port" . | quote }}
{{- if .Values.omniArgs.vaeUseSlicing }}
- "--vae-use-slicing"
{{- end }}
{{- if .Values.omniArgs.vaeUseTiling }}
- "--vae-use-tiling"
{{- end }}
{{- if .Values.omniArgs.enableCpuOffload }}
- "--enable-cpu-offload"
{{- end }}
{{- if .Values.omniArgs.numGpus }}
- "--num-gpus"
- {{ .Values.omniArgs.numGpus | quote }}
{{- end }}
{{- if .Values.omniArgs.stageConfigsPath }}
- "--stage-configs-path"
- {{ .Values.omniArgs.stageConfigsPath | quote }}
{{- end }}
{{- if and .Values.omniArgs.cacheBackend (ne .Values.omniArgs.cacheBackend "none") }}
- "--cache-backend"
- {{ .Values.omniArgs.cacheBackend | quote }}
{{- end }}
{{- if .Values.omniArgs.defaultSamplingParams }}
- "--default-sampling-params"
- {{ .Values.omniArgs.defaultSamplingParams | quote }}
{{- end }}
{{- if and .Values.omniArgs.workerBackend (ne .Values.omniArgs.workerBackend "multi_process") }}
- "--worker-backend"
- {{ .Values.omniArgs.workerBackend | quote }}
{{- end }}
{{- range .Values.omniArgs.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Define HuggingFace environment variables
*/}}
{{- define "chart.hf-env" -}}
- name: HF_HOME
  value: "/cache/huggingface"
- name: HOME
  value: "/cache"
{{- if .Values.hfToken }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-hf-token
      key: token
{{- end }}
{{- end }}

{{/*
Define ports for the pods
*/}}
{{- define "chart.container-port" -}}
{{-  default "8000" .Values.containerPort }}
{{- end }}

{{/*
Define service name
*/}}
{{- define "chart.service-name" -}}
{{-  if .Values.serviceName -}}
{{ .Values.serviceName | lower | trim }}
{{-  else -}}
{{ printf "%s-service" .Release.Name }}
{{-  end -}}
{{- end }}

{{/*
Define service port
*/}}
{{- define "chart.service-port" -}}
{{-  if .Values.servicePort }}
{{-    .Values.servicePort }}
{{-  else }}
{{-    include "chart.container-port" . }}
{{-  end }}
{{- end }}

{{/*
Define service port name
*/}}
{{- define "chart.service-port-name" -}}
"service-port"
{{- end }}

{{/*
Define container port name
*/}}
{{- define "chart.container-port-name" -}}
"container-port"
{{- end }}

{{/*
Define deployment strategy
*/}}
{{- define "chart.strategy" -}}
strategy:
{{-   if not .Values.deploymentStrategy }}
  type: Recreate
{{-   else }}
{{      toYaml .Values.deploymentStrategy | indent 2 }}
{{-   end }}
{{- end }}

{{/*
Define additional ports
*/}}
{{- define "chart.extraPorts" }}
{{-   with .Values.extraPorts }}
{{      toYaml . }}
{{-   end }}
{{- end }}

{{/*
Define chart external ConfigMaps and Secrets
*/}}
{{- define "chart.externalConfigs" -}}
{{-   with .Values.externalConfigs -}}
{{      toYaml . }}
{{-   end }}
{{- end }}

{{/*
Define startup, liveness and readiness probes
*/}}
{{- define "chart.probes" -}}
{{-   if .Values.startupProbe  }}
startupProbe:
{{-     with .Values.startupProbe }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{-   if .Values.readinessProbe  }}
readinessProbe:
{{-     with .Values.readinessProbe }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{-   if .Values.livenessProbe  }}
livenessProbe:
{{-     with .Values.livenessProbe }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{- end }}

{{/*
Define resources
*/}}
{{- define "chart.resources" -}}
requests:
  memory: {{ required "Value 'resources.requests.memory' must be defined !" .Values.resources.requests.memory | quote }}
  cpu: {{ required "Value 'resources.requests.cpu' must be defined !" .Values.resources.requests.cpu | quote }}
  {{- if and (gt (int (index .Values.resources.requests "nvidia.com/gpu")) 0) (gt (int (index .Values.resources.limits "nvidia.com/gpu")) 0) }}
  nvidia.com/gpu: {{ required "Value 'resources.requests.nvidia.com/gpu' must be defined !" (index .Values.resources.requests "nvidia.com/gpu") }}
  {{- end }}
limits:
  memory: {{ required "Value 'resources.limits.memory' must be defined !" .Values.resources.limits.memory | quote }}
  cpu: {{ required "Value 'resources.limits.cpu' must be defined !" .Values.resources.limits.cpu | quote }}
  {{- if and (gt (int (index .Values.resources.requests "nvidia.com/gpu")) 0) (gt (int (index .Values.resources.limits "nvidia.com/gpu")) 0) }}
  nvidia.com/gpu: {{ required "Value 'resources.limits.nvidia.com/gpu' must be defined !" (index .Values.resources.limits "nvidia.com/gpu") }}
  {{- end }}
{{- end }}

{{/*
Define user for the main container
*/}}
{{- define "chart.user" }}
{{-   if .Values.image.runAsUser  }}
runAsUser:
{{-     with .Values.image.runAsUser }}
{{-       toYaml . | nindent 2 }}
{{-     end }}
{{-   end }}
{{- end }}

{{/*
Define chart labels
*/}}
{{- define "chart.labels" -}}
{{-   with .Values.labels -}}
{{      toYaml . }}
{{-   end }}
{{- end }}
