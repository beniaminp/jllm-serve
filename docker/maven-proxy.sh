#!/bin/sh
# Prints the Maven arguments for a build behind an HTTP(S) proxy: Docker passes HTTP_PROXY / HTTPS_PROXY / NO_PROXY
# to builds, Maven only reads proxies from a settings file, so one is written to /tmp. Prints nothing without a proxy.
set -eu
url="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
[ -z "$url" ] && exit 0
rest="${url#*://}"
rest="${rest%%/*}"
user=""
pass=""
case "$rest" in
  *@*) cred="${rest%@*}"; rest="${rest##*@}"; user="${cred%%:*}"; pass="${cred#*:}" ;;
esac
host="${rest%:*}"
port="${rest##*:}"
[ "$port" = "$rest" ] && port=80
noproxy="$(printf '%s' "${NO_PROXY:-${no_proxy:-}}" | tr ',' '|')"
auth=""
[ -n "$user" ] && auth="<username>$user</username><password>$pass</password>"
cat > /tmp/maven-proxy-settings.xml <<XML
<settings><proxies>
<proxy><id>https</id><active>true</active><protocol>https</protocol><host>$host</host><port>$port</port>$auth<nonProxyHosts>$noproxy</nonProxyHosts></proxy>
<proxy><id>http</id><active>true</active><protocol>http</protocol><host>$host</host><port>$port</port>$auth<nonProxyHosts>$noproxy</nonProxyHosts></proxy>
</proxies></settings>
XML
printf '%s' "-s /tmp/maven-proxy-settings.xml"
