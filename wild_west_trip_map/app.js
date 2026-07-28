const map = L.map("map").setView([44.85, -110.25], 8);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

let activeLayers = [];
let segmentLines = [];
let stopMarkers = [];

const normalLineStyle = {
  weight: 5,
  color: "#2563eb",
  opacity: 0.75
};

const activeLineStyle = {
  weight: 8,
  color: "#f97316",
  opacity: 1
};

const dayMeta = {
  fri_sep_4: {
    mainWin: "Get west after work without overcomplicating the night.",
    risk: "Late departure, Chicago-area traffic, driver fatigue.",
    cutFirst: "Any extra stop beyond one combined dinner/fuel/bathroom stop."
  },

  sat_sep_5: {
    mainWin: "Finish the long Wisconsin Dells to Bismarck transport day cleanly.",
    risk: "Long mileage and stop creep.",
    cutFirst: "Any sightseeing or city detour."
  },

  sun_sep_6: {
    mainWin: "Reach Gardiner, check in, fuel, and be ready for Yellowstone.",
    risk: "Arriving too late and still forcing Mammoth.",
    cutFirst: "Mammoth Lower Terraces."
  },

  mon_sep_7: {
    mainWin: "Lamar Valley wildlife window plus Yellowstone Canyon.",
    risk: "Late wake-up, wildlife traffic, trying too many Canyon viewpoints.",
    cutFirst: "Extra Canyon viewpoint. If waking late, cut Lamar."
  },

  tue_sep_8: {
    mainWin: "Old Faithful and Grand Prismatic Overlook.",
    risk: "Parking delay at Grand Prismatic or waiting too long for geysers.",
    cutFirst: "West Thumb, then extra geyser-boardwalk time."
  },

  wed_sep_9: {
    mainWin: "Easy Grand Teton scenic transfer without draining energy.",
    risk: "Adding hikes or waiting too long for perfect photos.",
    cutFirst: "Mormon Row, then Snake River Overlook."
  },

  thu_sep_10: {
    mainWin: "Oxbow Bend sunrise, then safely reach Wall.",
    risk: "Bonus stops before a very long drive.",
    cutFirst: "Anything besides Oxbow Bend sunrise."
  },

  fri_sep_11: {
    mainWin: "Quick Badlands scenic drive, then reach Wisconsin Dells.",
    risk: "Turning Badlands into a hiking day.",
    cutFirst: "Ben Reifel Visitor Center, then Big Badlands Overlook if behind."
  },

  sat_sep_12: {
    mainWin: "Get home early enough to protect Sunday rest.",
    risk: "Chicago traffic and final-day fatigue.",
    cutFirst: "Any bonus stop."
  }
};


function clearMap() {
  activeLayers.forEach(layer => map.removeLayer(layer));
  activeLayers = [];
  segmentLines = [];
  stopMarkers = [];
}

function focusStop(coords) {
  map.setView(coords, Math.max(map.getZoom(), 10), {
    animate: true
  });
}

function focusSegment(segment) {
  map.fitBounds([segment.from, segment.to], {
    padding: [80, 80],
    animate: true
  });
}

function refreshMapSize() {
  setTimeout(() => {
    map.invalidateSize();
  }, 150);
}

function renderDayDetails(day, dayKey) {
  const meta = dayMeta[dayKey] || {};
  const startStop = day.stops[0]?.name || "Unknown";
  const endStop = day.stops[day.stops.length - 1]?.name || "Unknown";

  document.getElementById("details").innerHTML = `
    <div class="summary-card">
      <h2>${day.title}</h2>

      <div class="quick-grid">
        <div class="quick-item">
          <span class="quick-label">Start</span>
          <span class="quick-value">${startStop}</span>
        </div>

        <div class="quick-item">
          <span class="quick-label">End</span>
          <span class="quick-value">${endStop}</span>
        </div>

        <div class="quick-item">
          <span class="quick-label">Overnight</span>
          <span class="quick-value">${day.overnight}</span>
        </div>

        <div class="quick-item">
          <span class="quick-label">Stops / Sections</span>
          <span class="quick-value">${day.stops.length} / ${day.segments.length}</span>
        </div>
      </div>

      <div class="detail-row">
        <div class="detail-label">Main win</div>
        <div class="detail-value do">${meta.mainWin || day.summary}</div>
      </div>

      <div class="detail-row">
        <div class="detail-label">Risk</div>
        <div class="detail-value">${meta.risk || "Normal travel delay risk."}</div>
      </div>

      <div class="detail-row">
        <div class="detail-label">Cut first</div>
        <div class="detail-value skip">${meta.cutFirst || "Optional stops."}</div>
      </div>

      <p class="summary-text">${day.summary}</p>
    </div>

    ${renderSidebarLists(day)}
  `;

  attachSidebarEvents(day);
}

function renderStopButtons(day, activeIndex = null) {
  return day.stops.map((stop, index) => `
    <button class="stop-button ${index === activeIndex ? "active" : ""}" data-stop-index="${index}">
      ${index + 1}. ${stop.name}
    </button>
  `).join("");
}

function renderSegmentButtons(day, activeIndex = null) {
  return day.segments.map((segment, index) => `
    <button class="segment-button ${index === activeIndex ? "active" : ""}" data-segment-index="${index}">
      ${index + 1}. ${segment.name}
    </button>
  `).join("");
}

function attachSidebarEvents(day) {
  document.querySelectorAll(".stop-button").forEach(button => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.stopIndex);
      showStopDetails(day, index);
    });
  });

  document.querySelectorAll(".segment-button").forEach(button => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.segmentIndex);
      showSegmentDetails(day, index);
    });
  });
}

function renderSidebarLists(day, activeStopIndex = null, activeSegmentIndex = null) {
  return `
    <div class="stop-list">
      <h3 class="sidebar-section-title">Stops</h3>
      ${renderStopButtons(day, activeStopIndex)}
    </div>

    <div class="segment-list">
      <h3 class="sidebar-section-title">Route sections</h3>
      ${renderSegmentButtons(day, activeSegmentIndex)}
    </div>
  `;
}

function showSegmentDetails(day, index) {
  const segment = day.segments[index];

  focusSegment(segment);

  segmentLines.forEach(line => {
    line.setStyle(normalLineStyle);
  });

  segmentLines[index].setStyle(activeLineStyle);
  segmentLines[index].bringToFront();

  document.getElementById("details").innerHTML = `
    <div class="detail-card">
      <h2>${index + 1}. ${segment.name}</h2>

      <div class="chip-row">
        <span class="chip">Time: ${segment.time}</span>
        <span class="chip">Buffer: ${segment.buffer}</span>
      </div>

      <div class="detail-row">
        <div class="detail-label">Do</div>
        <div class="detail-value do">${segment.do}</div>
      </div>

      <div class="detail-row">
        <div class="detail-label">Skip / Rule</div>
        <div class="detail-value skip">${segment.skip}</div>
      </div>

      <div class="detail-row">
        <div class="detail-label">Notes</div>
        <div class="detail-value">${segment.detail}</div>
      </div>
    </div>

    ${renderSidebarLists(day, null, index)}
  `;

  attachSidebarEvents(day);
}

function showStopDetails(day, index) {
  const stop = day.stops[index];

  focusStop(stop.coords);

  if (stopMarkers[index]) {
    stopMarkers[index].openPopup();
  }

  segmentLines.forEach(line => {
    line.setStyle(normalLineStyle);
  });

  document.getElementById("details").innerHTML = `
    <div class="detail-card">
      <h2>${index + 1}. ${stop.name}</h2>

      <div class="detail-row">
        <div class="detail-label">Stop note</div>
        <div class="detail-value">${stop.note}</div>
      </div>
    </div>

    ${renderSidebarLists(day, index, null)}
  `;

  attachSidebarEvents(day);
}

function populateDayDropdown() {
  const daySelect = document.getElementById("daySelect");

  Object.entries(tripData).forEach(([dayKey, day]) => {
    const option = document.createElement("option");
    option.value = dayKey;
    option.textContent = day.title;
    daySelect.appendChild(option);
  });
}

function showDay(dayKey) {
  clearMap();

  const day = tripData[dayKey];
  const boundsPoints = [];

  renderDayDetails(day, dayKey);

    day.stops.forEach((stop, index) => {
    const numberIcon = L.divIcon({
        className: "numbered-marker",
        html: `<div>${index + 1}</div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
        popupAnchor: [0, -15]
    });

    const marker = L.marker(stop.coords, {
        icon: numberIcon
    })
        .addTo(map)
        .bindPopup(`
        <strong>${index + 1}. ${stop.name}</strong><br>
        ${stop.note}
        `);

    marker.on("click", () => {
        showStopDetails(day, index);
    });
    
    stopMarkers.push(marker);
    activeLayers.push(marker);
    boundsPoints.push(stop.coords);
    });

  day.segments.forEach((segment, index) => {
  const line = L.polyline([segment.from, segment.to], normalLineStyle).addTo(map);

  line.bindPopup(`<strong>${index + 1}. ${segment.name}</strong><br>${segment.time}`);

  line.on("click", () => {
    showSegmentDetails(day, index);
  });

  segmentLines.push(line);
  activeLayers.push(line);
  boundsPoints.push(segment.from, segment.to);
});

  map.fitBounds(boundsPoints, {
  padding: [40, 40]
  });

  refreshMapSize();
}

populateDayDropdown();

document.getElementById("daySelect").addEventListener("change", event => {
  showDay(event.target.value);
});

showDay(Object.keys(tripData)[0]);
window.addEventListener("resize", () => {
  refreshMapSize();
});